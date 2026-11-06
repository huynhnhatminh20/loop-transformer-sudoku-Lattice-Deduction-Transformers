"""Pure-torch AdamATan2, math-identical to upstream adam-atan2==0.0.3.

Why vendored: the PyPI package needs its CUDA extension
(adam_atan2_backend, csrc/*.cu: nvcc + torch headers) which is not
available in the Modal debian_slim build. The fused kernel implements
plain elementwise math (see csrc/adam_atan2.cu lines 28-60), so this
module replays the SAME formulas with torch ops (compute in fp32
opmath, store back in param dtype — like the kernel's opmath_type).
Only fp-associativity (fma vs mul-add order) can differ at ~1-ulp.

Formulas per element (step already incremented, bc1 = 1-b1^step):
    step_size = lr / bc1
    wd_alpha  = 1 - lr * wd
    param    *= wd_alpha
    m += (1-b1) * (grad - m)
    v += (1-b2) * (grad*grad - v)
    denom = sqrt(v) / sqrt(1-b2^step)
    param -= step_size * atan2(m, denom)

Matches: upstream pretrain.py `AdamATan2(params, lr=0, weight_decay,
betas)` call protocol (lr set per-step by scheduler).
"""

import math

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import List, Tuple, Union


class AdamATan2(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:  # type: ignore
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        hook = getattr(self, "_cuda_graph_capture_health_check",
                       getattr(self, "_accelerator_graph_capture_health_check",
                               None))
        if hook is not None:
            hook()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            if isinstance(lr, Tensor):
                raise AssertionError("scalar lr only (upstream restriction)")
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamATan2 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros(
                        (), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"].add_(1)
                step = float(state["step"].item())
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                step_size = lr / bc1
                wd_alpha = 1.0 - lr * wd
                mb1 = 1.0 - beta1
                mb2 = 1.0 - beta2
                bc2_sqrt = math.sqrt(bc2)

                # opmath fully in fp32, stored back in param dtype
                # (kernel parity: compute in opmath_type, store scalar_type)
                with torch.no_grad():
                    p32 = p.float()
                    m32 = state["exp_avg"].float()
                    v32 = state["exp_avg_sq"].float()
                    g32 = grad.float()
                    p32.mul_(wd_alpha)
                    m32.add_(g32.sub(m32), alpha=mb1)
                    v32.add_(g32.mul(g32).sub(v32), alpha=mb2)
                    denom = v32.sqrt().div_(bc2_sqrt)
                    p32.add_(torch.atan2(m32, denom), alpha=-step_size)
                    p.copy_(p32)
                    state["exp_avg"].copy_(m32)
                    state["exp_avg_sq"].copy_(v32)
