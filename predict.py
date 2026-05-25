import torch
from PIL import Image
from torchvision import transforms
from model import get_model
from gradcam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = get_model()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🔥 GradCAM (DenseNet feature extractor)
gradcam = GradCAM(model, model.features)

def predict(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)

    return probs, img_tensor


def get_heatmap(img_tensor, class_idx):
    return gradcam.generate(img_tensor, class_idx)