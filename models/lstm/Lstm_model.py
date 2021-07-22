
import torch
from torch.nn.modules.dropout import Dropout



class LSTM(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_dim,drop=0.1,batch_first=True):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first)
		self.linear = torch.nn.Linear(hidden_size,hidden_size)
		self.act = torch.nn.Tanh()
		self.linear2 = torch.nn.Linear(hidden_size,output_dim)
		self.act2 = torch.nn.Tanh()

	def forward(self, x):

		output,_ = self.lstm(x)
		output = self.linear(output)
		output = self.act(output)
		output = self.linear2(output)
		output = self.act2(output)

		return output

	# maybe add one more layer a