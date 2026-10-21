import json
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.domain.models import EvaluationMetrics, PairedSimulationResult
from src.repositories.dataset_repository import DatasetRepository
from src.services.resnet_builder import LinearWarmupCosineAnnealingLR, create_resnet50

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orquestrador do estudo empírico com 30 simulações pareadas."""

    def __init__(
        self,
        seeds: list[int] | None = None,
        epochs_per_simulation: int = 5,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.seeds = seeds or list(range(1001, 1031))  # N = 30
        self.epochs = epochs_per_simulation
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.repo = DatasetRepository()

    def train_and_evaluate(
        self, pretrained: bool, seed: int, train_loader, test_loader
    ) -> EvaluationMetrics:
        """Treina uma ResNet-50 sob a semente fixada e avalia no teste."""
        model = create_resnet50(pretrained=pretrained, seed=seed).to(self.device)
        criterion = nn.CrossEntropyLoss()

        if pretrained:
            # Transfer Learning: treina a cabeça e ajusta blocos profundos com taxa moderada
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = None
        else:
            # Scratch: treinamento a partir do zero com Warmup e Cosine Annealing
            optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=1,
                max_epochs=self.epochs,
                base_lr=3e-4,
                min_lr=1e-6,
            )

        model.train()
        for epoch in range(self.epochs):
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()

        # Avaliação determinística no conjunto de teste independente
        model.eval()
        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device, non_blocking=True)
                out = model(x)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())

        f1 = float(f1_score(all_targets, all_preds, average="macro"))
        acc = float(accuracy_score(all_targets, all_preds))
        return EvaluationMetrics(f1_macro=round(f1, 4), accuracy=round(acc, 4))

    def run_all_simulations(self) -> list[PairedSimulationResult]:
        """Executa as 30 simulações pareadas."""
        results: list[PairedSimulationResult] = []
        test_loader = self.repo.get_test_loader(batch_size=self.batch_size)

        logger.info(
            f"Iniciando {len(self.seeds)} simulações pareadas no dispositivo: {self.device}"
        )

        for seed in tqdm(self.seeds, desc="Simulações Pareadas (N=30)"):
            train_loader = self.repo.get_train_loader(
                batch_size=self.batch_size, seed=seed, shuffle=True
            )

            # Cenário 1: Transfer Learning (ImageNet)
            metrics_s1 = self.train_and_evaluate(
                pretrained=True,
                seed=seed,
                train_loader=train_loader,
                test_loader=test_loader,
            )

            # Cenário 2: Scratch com Warmup + Cosine Annealing
            metrics_s2 = self.train_and_evaluate(
                pretrained=False,
                seed=seed,
                train_loader=train_loader,
                test_loader=test_loader,
            )

            paired = PairedSimulationResult(
                seed=seed, scenario_1=metrics_s1, scenario_2=metrics_s2
            )
            results.append(paired)

        return results

    def save_results(
        self,
        results: list[PairedSimulationResult],
        filepath: str | Path = "results/simulacoes_pareadas.json",
    ) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Simulações salvas em {path}")
