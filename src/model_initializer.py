import torch

from network import Network

class ModelInitializer:
    def __init__(self, **kwargs):
        self.device = kwargs['device'] if 'device' in kwargs else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.network = Network().to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.CrossEntropyLoss()