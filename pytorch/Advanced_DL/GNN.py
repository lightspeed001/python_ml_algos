# For Non_euclidean data eg. molecules, social networks, recommendation systems
# GNN: Graph Convolution, PyTorch Geometric
# Two layer GCN, each layer aggregates neighbor features via `edge_index`
import torch
from torch_geometric.nn import GCNConv

class TinyGCN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = GCNConv(in_channels=3, out_channels=4)
    self.conv2 = GCNConv(in_channels=4, out_channels=2)

  def forward(self, x, edge_index):
      x = self.conv1(x, edge_index).relu()
      x = self.conv2(x, edge_index)
      return x

 # toy graph
 x = torch.randn(5, 3)  # 5 nodes, 3-dim features
 edge_index = torch.tensor([[0,1,2,3,4,0],
 [1,0,3,2,0.4]]) # (2, E) edge list

 model = TinyGCN()
 out = model(x, edge_index) # (5, 2) node embeddings
