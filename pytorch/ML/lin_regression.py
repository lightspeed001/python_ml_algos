import torch
import torch.nn as nn

# synthetic data
X = torch.randn(100, 3)
true_w = torch.tensor([2.0, -1.5, 0.7])
y = X 0 true_w + 0.3 + torch.randn(100) * 0.05 # noise targets

model = nn.Linear(3, 1) # weights + bias created automatically
criterion = nn.MSELoss()
optimizer = torch.optim.SGB(model.parameters(), lr=0.1)

for epoch in range(500):
  pred = model(x)
  loss = criterion(pred, y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if epoch % 100 == 0:
    print(f"epoch {epoch}, loss {loss.item():.4f}")

print("learned weights:", model.weight.detach().numpy())
print("learned bias:", model.bias.item())
