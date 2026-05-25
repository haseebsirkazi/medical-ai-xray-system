import torch
import torch.nn.functional as F
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, x, class_idx):

        activations = {}

        # ✔ only forward hook (safe)
        def hook_fn(module, input, output):
            activations["value"] = output

        handle = self.target_layer.register_forward_hook(hook_fn)

        # forward pass
        output = self.model(x)

        handle.remove()

        # get activations
        act = activations["value"][0]  # [C,H,W]

        # get score
        score = output[0, class_idx]

        # backward (no hooks)
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # ❗ SAFE fallback: use activations (NOT grads)
        weights = torch.mean(act, dim=(1, 2))

        cam = torch.zeros(act.shape[1:], device=x.device)

        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        return cam