import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloader(batch_size=64):
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.0, 0.0, 0.0), (255.0, 255.0, 255.0))
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5)
        ])
    
    test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    trainset = torchvision.datasets.CIFAR10(root=r'D:\Datasets\CIFAR-10', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=r'D:\Datasets\CIFAR-10', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

