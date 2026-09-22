# Continuos depth models eg. irregular time series or physics informed ML
# Defines a continous-time dynamics dy/dt = -y and integrates it with `odeint`
import torch
from torchdiffeq import odeint

class ODEFunc(torch.nn.Module):
  def forward(self, t, y):
    # dy/dt = -y (simple exponential decay)
    return -y

y0 = torch.tensor([1.0]) # initial condition
t = torch.linspace(0., 5., steps=50)  # integration times
sol = odeint(ODEFunc(), y0, t)  # (len(t), 1)

# sol = exp(-t)
