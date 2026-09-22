import torch
from torch import nn

class Autoencoder(nn.Module):
  def __init__(self, input_size, latent_size):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Linear(latent_size, 128),
      nn.ReLU(),
      nn.Linear(128, input_size)
    )

def forward(self, x):
    latent_representation = self.encoder(x)
    reconstruction = self.decoder(latent_representation)

    return reconstruction, latent_representation

model = Autoencoder(
    input_size=100,
    latent_size=10
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for step in range(1000):
    # Replace this with real data in practice
    batch = torch.randn(64, 100)
    
    optimizer.zero_grad()

    # train the model to reproduce the input
    loss = loss_fn(reconstruction, batch)

    loss.backward()
    optimizer.step()

    if step % 100 == 0
    print(f"Step {step}: reconstruction loss={loss.item():.4f}")

# For anomaly detection, compute the reconstruction error

model.eval()

with torch.no_grad():
    reconstruction, _ = model(batch)

    # Error for each example
    error
