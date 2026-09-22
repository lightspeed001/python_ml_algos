# used primarily for image/video generation
# Shows the core idea: a network predicts noise conditioned on a timestamp, then iteratively denoises
import torch, torch.nn as nn

class TinyDiffusion(nn.Module):
  def __init__(self, timesteps=10):
    super().__init__()
    self.timesteps = timesteps
    self.net = nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

  def forward(self, x, t):
      # x: (batch, 1) noisy image; t: integer timestep
      t_norm = t.float() / self.timesteps # scale timestep to [0,1]
      return self.net(torch.cat([x, t_norm.unsqueeze(-1)], dim=1))

# sampling (very crude)
 def sample(model, shape, device):
     x = torch.randn(shape, device=device)
     for t in reversed(range(model.timesteps)):
         eps = model(x, torch.full((shape[0],), t, device=device))
         x = x - eps * 0.1
         return x
