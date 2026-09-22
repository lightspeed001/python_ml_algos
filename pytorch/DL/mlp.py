import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data
x, y = make_classification(
  n_samples=3000,
  n_features=20,
  n_features=20,
  n_informative=12,
  random_state=42
)

# Normalize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)


X-train, X_test, y_train, y_test = train_test_split(
  X, y,
  test_size=0.2,
  random_state=42,
  stratify=y
)

# Convert NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
  TensorDataset(X_train, y_train),
  batch_size=64,
  shuffle=True
)

class MLP(nn.Module):
  def __init__(self, input_size, num_classes):
    super().__init__()

    self.network = nn.Sequential(
      nn.Linear(input_size, 128),
      nn.ReLU(),
      # Dropout can reduce overfitting
      nn.Dropout(0.2),
      nn.Linear(128, 64),
      nn.ReLU(),

      nn.Linear(64, num_classes)
    )

  def forward(self, x):
    # The final layer returns raw logits.
    return self.network(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLP(input_size=20, num_classes=2).to(device)

    # CrossEntropyLoss expects integer class labels
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
      model.parameters(),
      lr=1e-3
    )

    for epoch in range(20):
      model.train()
      total_loss = 0

      for batch_X, batch_y in train_loader:
        batch_x = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate after each epoch
    model.eval()
    with torch.no_grad():
      logits = model(X_test.to(device))
      predictions = logits.argmax(dim=1)
      accuracy = (predictions == y_test.to(device)).float().mean()

    print(
      f"Epoch {epoch + 1:02d} | "
      f"loss={total_loss / len(train_loader):.4f} | "
      f"accurracy={accuracy.item():.3f}"
    )
