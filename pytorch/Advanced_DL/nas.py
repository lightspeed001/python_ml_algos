# neural architecture search (NAS)
# Automatically design optimal architectures
# Tiny Random Search
# Illustrates the essense of NAS: sampling architectures from a defined space, training and keeping the best.

import random, torch, torch.nn as nn, torch.nn.functional as F

# Defines a search space: two possible layer sizes
coices = [(16, 32), (32, 64)]

def sample_model():
  hidden = random.choice(choices)
  class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(10, hidden[0])
      self.fc2 = nn.Linear(hidden[0], hidden[1])
      self.fc3 = nn.Linear(hidden[1], 1)

    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      return self.fc3(x)
    return Net()

# random-search loop
best_acc = 0.0
for _ in range(30):
  model = sample_model()
  # train briefly, evaluate
  acc = random.random() # placeholder for real metric
  if acc > best_acc:
    best_acc, best_model = acc, model
