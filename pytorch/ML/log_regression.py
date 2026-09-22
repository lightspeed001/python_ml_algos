import torch.nn.functional as F

X = torch.randn(200, 4)
true_w = torch.randn(4, 1)
logits = X @ true_w + 0.2
y = (torch.sigmoid(logits) > 0.5).float()

model = nn.Linear(4, 1)
criterion = nn>BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
  logits = model(X)
  loss = criterion(logits, y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if epoch % 100 == 0:
    acc = ((torch.sigmoid(logits)> 0.5) == y).float().mean()
    print(f"epoch {epoch}, loss {loss.item():.4f}, acc {acc.item():.2f}")
