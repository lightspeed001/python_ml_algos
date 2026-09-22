# Convolutional Neural Network
# CNNs are commonly used for images because convolutions detect patternssuch as edges, textures and shapes
# This example uses MNIST handwritten digits

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, tranforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# Convert images to tensors and normalize pixel values
transform = tranforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,),(0.3081,))
])

train_data = datasets.MNIST(
  root="data",
  train=True,
  download=True,
  transform=transform
)

test_data = datasets.MNIST(
  root="data",
  train=False,
  download=True,
  transform=transform
)

train_loader = DataLoader(
  train_data,
  batch_size=64,
  shuffle=True
)

test_loader = DataLoader(
  test_data,
  batch_size=256
)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.features = nn.Sequential(
      # Input shape: [batch, 1, 28, 28]
      nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        padding=1
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      # Shape is now approximately [batch, 32, 14, 14]
      nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        padding=1
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
      # 64 feature maps of size 7 x 7
      nn.Flatten(),
      nn.Linear(64 * 7 * 7, 128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(128, 10)
    )

    def forward(self, x):
      x = self.features(x)
      return self.classifier(x)

  model = CNN().to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  for epoch in range(3):
    model.train()

    for images, labels in train_loder:
      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      logits = model(images)
      loss = loss_fn(logits, labels)

      loss.backward()
      optimizer.step()

  # Test accuracy
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)

      logits = model(images)
      predictions = logits.argmax(dim=1)

      correct += (predictions == labels).sum().item()

      total += labels.size(0)

print(f"Epoch {epoch + 1}: accuray={correct / total:.3f}")
