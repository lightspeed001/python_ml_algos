# Transformer encoder for sequence classification
# Transformers are widely used for language, time series, and other sequence problems. Unlike an LSTM, a Transformer can process sequence positions in prallel.
import torch
from torch import nn


class TransformerClassification(nn.Module){
  def __init__(
    self,
    input_size,
    model_size,
    num_heads,
    num_layers,
    num_classes,
    max_sequence_length
  ):
    super().__init__()

  # Project input features into the TRansformer dimention
  self.input_projection = nn.Linear(input_size, model_size)

  # Learnable positional embeddings
  self.position_embedding = nn.Parameter(
    torch.randn(1, max_sequence_length, model_size)
  )

  encoder_layer = nn.TransformerEncoderLayer(
    d_model=model_size,
    nhead=num_heads,
    dim_feedforward=model_size * 4,
    dropout=0.1,
    batch_first=True,
    activation="gelu"
  )

  self.encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=num_layers
  )

  self.classifier = nn.Linear(model_size, num_classes)

def forward(self, x):
  # x shape: [batch, sequence_length, input_size]
  x = self.input_projection(x)

  sequence_length = x.size(1)
  x = x + self.position_embedding[:, :sequence_length, :]

  encoded = self.encoder(x)

  # Mean-pool over the sequence
  pooled = encoded.mean(dim=1)
  return self.classifier(pooled)

model = TransformerClassifier(
  input_size=5,
  model_size=64,
  num_heads=4,
  num_layers=2,
  num_classes=2,
  max_sequence_length=30
)

example_input =  torch.randn(0, 30, 5)
logits = model(example_input)

print(logits.shape)
# output: torch.Size([8, 2])
}

# training step for Transformer is the same pattern as MLP and LSTM
# labels = torch.randint(0, 2, (8,))
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# optimizer.zero_grad()

# logits = model(example_input)
# loss = loss_fn(logits, labels)

# loss.backward()
# optimizer.step()
