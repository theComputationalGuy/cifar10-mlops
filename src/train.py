import os
from tqdm import tqdm
import torch

from model_initializer import ModelInitializer
from train_one_epoch import train_one_epoch

def train(dataloader, classes, **kwargs):
    save_path = kwargs['save_path'] if 'save_path' in kwargs else None

    model = ModelInitializer()
    
    start_epoch = 0
    num_epochs = kwargs['num_epochs']

    if save_path is not None and os.path.exists(save_path):
        print(f"Loading model from {save_path}")
        # Load the model here
        checkpoint = torch.load(save_path)
        
        start_epoch = checkpoint['epoch']
        model.network.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("No existing model found, starting from scratch.")

    for epoch in tqdm(range(start_epoch, num_epochs)):
        train_one_epoch(model, dataloader)

        if save_path is not None:
            torch.save({
                'epoch': epoch + 1,
                'model': model.network.state_dict(),
                'optimizer': model.optimizer.state_dict()
            }, save_path)