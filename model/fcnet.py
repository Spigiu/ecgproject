import torch.nn as nn


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_len, num_classes, hidden_units, hidden_layers=2):
        super(FullyConnectedNetwork, self).__init__()
        assert hidden_layers > 0, "At least one hidden layer required"

        layer_list = [nn.Flatten(), nn.Linear(input_len, hidden_units), nn.ReLU()]
        
        
        for i in range(1, hidden_layers):
            if i==1:
                layer_list.extend([nn.Linear(hidden_units // (2*i-1), hidden_units // (2*i)), nn.ReLU(),nn.Dropout(p=0.8)])
            else:
                layer_list.extend([nn.Linear(hidden_units // (2*i-2), hidden_units // (2*i)), nn.ReLU(),nn.Dropout(p=0.8)])
        
        layer_list.append(nn.Linear(hidden_units // (2*i), num_classes, nn.Softmax(dim=1)))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
