import torch
from torch import nn



class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_dim,bidirectional = False,dropout=0.1):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional = bidirectional,dropout = dropout)
		if bidirectional:
			hidden_size *=2
			self.num_layers *=2
		self.linear = nn.Linear(hidden_size, output_dim)

	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

		out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

		out = self.linear(out[:, -1, :])

		return out
