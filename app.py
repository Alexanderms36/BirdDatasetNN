import torch
from load import load_dataset
from model import CNN


classes = ['goose', 'turkey', 'chicken', 'rooster', 'ostrich', 'duck', 'chick']
model = CNN()
model.load_state_dict(torch.load("model_weights.pth"))

while True:
        print("==============================================")
        print("Type a picture path (or type 'quit' to close)")
        img_path = input()
        if (img_path == "quit"):
                print("App closed.")
                break
        try:
                image = load_dataset(img_path, is_dataset=False)

                model.eval()
                with torch.no_grad():
                        outputs = model(image)
                        _, pred_class = torch.max(outputs, 1)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1).view(-1)

                pred_index = pred_class.item()
                pred_prob = probabilities[pred_index] * 100

                print(f"It's a {classes[pred_index]}")
                print(f"Probability: {pred_prob:.2f}%")
        except:
                print("image path is incorrect")

