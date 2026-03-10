import torch
import torch.nn as nn
import torch.nn.functional as F
from work.gnss.encoder import GNSSFeatMapEncoder as GNSSEncoder

class Decoder(nn.Module):
	def __init__(self, input_dim=128, hidden_dim=64, output_dim=1):
		super(Decoder, self).__init__()

		# 1. Attention Pooling
		self.attention = nn.Linear(input_dim, 1)

		# 2. MLP
		self.mlp = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.3)
		)

		# 3. Regression
		self.regression = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):

		# 1. Attention Pooling
		scores = self.attention(x).squeeze(-1)
		weights = F.softmax(scores, dim=1).unsqueeze(-1)

		context = torch.sum(x * weights, dim=1)

		# 2. MLP
		features = self.mlp(context)

		# 3. Regression
		output = self.regression(features)

		return output
	
class GNSSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GNSSEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x