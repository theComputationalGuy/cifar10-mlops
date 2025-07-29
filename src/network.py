import torch

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.network(x)
        return x