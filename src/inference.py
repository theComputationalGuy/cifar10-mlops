import torch
from torchvision import transforms

from src.model_initializer import ModelInitializer

def predict(model_path: str, input):
    model = ModelInitializer(device='cpu')
    checkpoint = torch.load(model_path)
    model.network.load_state_dict(checkpoint['model'])

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    input = transform(input).unsqueeze(0)
    # input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)

    model.network.eval()
    with torch.inference_mode():
        output = model.network(input)
        prediction = model.classes[torch.argmax(output, dim=1)[0]]

    return prediction