import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(pretrained: bool = True, seed: int = 1001) -> nn.Module:
    """Cria a arquitetura ResNet-50 configurada para 2 classes (NORMAL vs PNEUMONIA).

    Cenário 1: Transfer Learning com pesos ImageNet-1k pré-treinados.
    Cenário 2: Inicialização do zero (Scratch) com Kaiming Normal (He et al., 2015).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)
        # Inicialização He / Kaiming para convoluções com ReLU
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # Substituir a camada fully connected para 2 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model


class LinearWarmupCosineAnnealingLR:
    """Scheduler com Warmup Linear inicial seguido por decaimento de cosseno, atualizado por batch.

    Chamar step() após cada optimizer.step(). O lr já é ajustado na construção, de modo que o
    primeiro batch usa base_lr / warmup_steps (e não base_lr). O último batch usa min_lr.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
        self._apply_lr()

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step + 1) / self.warmup_steps
        cosine_steps = max(1, self.total_steps - self.warmup_steps - 1)
        progress = min(1.0, (self.current_step - self.warmup_steps) / cosine_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        self.current_step += 1
        self._apply_lr()

    def _apply_lr(self) -> None:
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
