from torchvision import datasets, transforms
from PIL import Image


def load_dataset(dir_path, img_size=160, is_dataset=True):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if (is_dataset):
        dataset = datasets.ImageFolder(root=dir_path, transform=transform)
        return dataset
    
    image = Image.open(dir_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image
