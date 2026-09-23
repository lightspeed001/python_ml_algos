# Scalable models with sparce activation eg. Googles Switch Transformers
import torch, torch.nn as nn, torch.nn.functional as F

class TinyMoE(nn.Module):
  def __init__(self, d_in=4, d_hidden=8, n_expert=3):
    super().__init__()
    self.experts = nn.ModuleList([nn.Sequential(
      nn.Linear(d_in, d_hidden), nn.ReLU(),
      nn.Linear(d_hidden, d_in)
    ) for _ in range(n_experts)])
    self.gate = nn.Linear(d_in, n_experts) # produces softmax weights

    def forward(self, x):
        # x: (batch, d_in)
        gate_weights = F.softmax(self.gate(x), dim=1) # (batch, n_experts)
        experts_outs = torch.stack([e(x) for e in self.experts], dim=1) # (batch, n_experts, d_in)
        out = torch.sum(gate_weights.unsqueeze(-1) * expert_outs, dim=1)
        return out

x = torch.randn(2, 4)
model = TinyMoE()
y = model(x) # (2, 4) weighted combination of expert outputs
