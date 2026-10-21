import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Implementação formal do Grad-CAM (Selvaraju et al., ICCV 2017) para ResNet-50."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        self._hook_handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(
        self, module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
        """Gera o mapa de calor normalizado [0, 1] para a classe alvo."""
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Falha ao capturar gradientes ou ativações da camada convolucional.")

        # Global Average Pooling dos gradientes (pesos alfa)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Combinação linear ponderada
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)  # Apenas ativações com influência positiva

        cam_np = cam.cpu().numpy()
        cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np) + 1e-8)
        cam_resized = cv2.resize(cam_np, (224, 224))
        return cam_resized

    def overlay_heatmap(
        self, raw_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5
    ) -> Image.Image:
        """Sobrepõe o mapa de calor colorido sobre a radiografia original em escala RGB."""
        img = raw_image.resize((224, 224)).convert("RGB")
        img_np = np.array(img)

        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_cam = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        overlaid = np.uint8(alpha * colored_cam + (1 - alpha) * img_np)
        return Image.fromarray(overlaid)

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
