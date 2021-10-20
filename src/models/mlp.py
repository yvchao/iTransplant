import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_n, out_n, h_n=20, layer_n=3):
        super(MLP, self).__init__()
        self.layer_n =layer_n
        
        
        if self.layer_n>1:
            self.input_layer = nn.Linear(in_n, h_n)
            self.middle_layers=nn.ModuleList([nn.Linear(h_n, h_n) for i in range(layer_n-2)])
            self.output_layer = nn.Linear(h_n,out_n)
            self.activation = nn.LeakyReLU()
        else:
            self.layer = nn.Linear(in_n, out_n)

    def forward(self, x):
        if self.layer_n>1:
            x = self.input_layer(x)
            x = self.activation(x)
            for i in range(len(self.middle_layers)):
                x = self.middle_layers[i](x)
                x = self.activation(x)
            x = self.output_layer(x)
        else:
            x = self.layer(x)
        return x