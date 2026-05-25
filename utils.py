from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    return loader, dataset.classes