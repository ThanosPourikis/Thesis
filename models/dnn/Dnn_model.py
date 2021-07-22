from torch import nn


class Dnn(nn.Module):
    def __init__(self,input_size,output_dim):
        super(Dnn,self).__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size,output_dim)
        )
    def forward(self,x):
        out = self.layer(x)
        return (out)



def run(df):
    input_size = len(df) - 1
    output_dim = 24
    model = Dnn(input_size,output_dim)
    out = model(df)
    