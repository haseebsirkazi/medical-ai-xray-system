import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.request
import numpy as np

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL URL (HUGGING FACE) ----------------
MODEL_URL = "https://huggingface.co/Haseebsirkazi/medical-xray-model/tree/main"
MODEL_PATH = "model.pth"

# ---------------- DOWNLOAD MODEL IF NOT PRESENT ----------------
if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading model from Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------------- MODEL ARCHITECTURE ----------------
model = models.densenet121(pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # FIX: ensures 3-channel input
    transforms.ToTensor(),
])

# ---------------- PREDICT FUNCTION ----------------
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)

    return probs.cpu().numpy(), image


# ---------------- GRAD-CAM SAFE VERSION ----------------
def get_heatmap(img_tensor, class_idx):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook last conv layer
    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    model.zero_grad()

    output = model(img_tensor)
    class_score = output[0, class_idx]

    class_score.backward()

    grads = gradients[0]
    acts = activations[0]

    # FIX: avoid inplace / view errors
    pooled_grads = torch.mean(grads, dim=(2, 3), keepdim=True)

    cam = (pooled_grads * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam.detach().cpu().numpy()