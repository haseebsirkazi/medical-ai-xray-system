import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.request

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL PATH + DOWNLOAD ----------------
MODEL_URL = "https://huggingface.co/Haseebsirkazi/medical-xray-model/resolve/main/model.pth"
MODEL_PATH = "model.pth"

# download if not exists
if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading model from Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------------- MODEL ARCHITECTURE ----------------
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.to(device)

# ---------------- SAFE MODEL LOADING ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # FIX: X-ray 1 channel → 3 channel
    transforms.ToTensor(),
])

# ---------------- PREDICTION FUNCTION ----------------
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)

    return probs.cpu().numpy(), image


# ---------------- GRAD-CAM ----------------
def get_heatmap(img_tensor, class_idx):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]

    # register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    model.zero_grad()

    output = model(img_tensor)
    score = output[0, class_idx]

    score.backward()

    grads = gradients[0]
    acts = activations[0]

    # Grad-CAM computation
    pooled_grads = torch.mean(grads, dim=(2, 3), keepdim=True)

    cam = (pooled_grads * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam.detach().cpu().numpy()