import torch
from load import load_dataset
from model import CNN


classes = ['goose', 'turkey', 'chicken', 'rooster', 'ostrich', 'duck', 'chick']
model = CNN()
model.load_state_dict(torch.load("model_weights.pth"))

print("Type a picture path")
img_path = input()

image = load_dataset(img_path, is_dataset=False)

model.eval()
with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

print(f"It's a {classes[predicted_class.item()]}")