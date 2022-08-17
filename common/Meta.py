import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, feature_size):
        super(MLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.features(x)
