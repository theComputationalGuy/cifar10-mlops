import torch

def train_one_epoch(model, dataloader):
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(model.device), labels.to(model.device)

        model.optimizer.zero_grad()
        
        outputs = model.network(inputs)
        loss = model.loss_fn(outputs, labels)
        
        loss.backward()
        model.optimizer.step()

        if i % 200 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")