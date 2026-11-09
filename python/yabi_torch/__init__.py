"""yabi-torch — a batched, differentiable ODE solver (BDF + continuous adjoint).

```python
import torch
from yabi_torch import solve_ivp

y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64, requires_grad=True)
yT = solve_ivp(f, y0, t_span=(0.0, 2.0), rtol=1e-8, atol=1e-10)  # automatic Jacobian
yT.backward(torch.ones_like(yT))   # y0.grad = d sum(yT) / d y0 via the adjoint
```
"""

from ._solver import solve_ivp
# Re-export the internals the test suite exercises directly.
from ._solver import _automatic_jacobian, _automatic_jvp, _wrap_rhs

__all__ = ["solve_ivp"]
