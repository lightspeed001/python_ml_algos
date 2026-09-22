# RNN: Used for very short sequential data. (Minimal compute requirements, Vanishing gradient/short memory)
# LSTM: Variant of RNNs used for long length sequential data with complex patterns (Superior Long term memory, High computational cost)
# GRUs: Variant of RNNs used for medium length sequential data with limited data (Faster training than LSTM, SLightly less expressive than LSTM)
# Single layer LSTM with one hidden unit dimension, followed by a linear read-out
import torch, torch.nn as nn

class TinyLSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(input_size=1, hidden_size=4, num_layers=1, batch_first=True)
    self.fc = nn.Linear(4, 1)

  def forward(self, x):
      # x: (batch, seq_len, 1)
      out, _ = self.lstm(x)  # out: (batch, seq_len, hiddden)
      out = out[:, -1, :]    # take last time-step
      return self.fc(out)    # (batch, 1)

 # example usage
 seq = torch.randn(2, 5, 1) # 2 sequences, length 5
 model = TinyLSTM()
 pred = model(seq) # (2, 1)



