import argparse

from dataloader import get_cifar10_dataloader
from train import train

def run(**kwargs):
    trainloader, testloader, classes = get_cifar10_dataloader(batch_size=kwargs['batch_size'])
    
    train(trainloader, classes, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the script with specified arguments.")

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=10, required=True)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    run(
        batch_size=args.batch_size,
        save_path=args.save_path,
        num_epochs=args.num_epochs
    )