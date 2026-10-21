import json
import logging
from pathlib import Path
from typing import Sequence

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
    """Orquestrador do estudo empírico com 30 simulações pareadas e persistência granular."""

    def __init__(
        self,
        seeds: Sequence[int] | None = None,
        epochs_per_simulation: int = 5,
        batch_size: int = 64,
        device: str | None = None,
        results_dir: str | Path = "results",
        checkpoint_seeds: Sequence[int] | None = None,
    ) -> None:
        self.seeds = list(seeds if seeds is not None else range(1001, 1031))  # N = 30
        self.epochs = epochs_per_simulation
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.repo = DatasetRepository()
        self.results_dir = Path(results_dir)
        self.runs_dir = self.results_dir / "runs"
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.consolidated_path = self.results_dir / "simulacoes_pareadas.json"

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Por padrão, salva checkpoint de modelo para a semente de referência 1001
        self.checkpoint_seeds = set(checkpoint_seeds if checkpoint_seeds is not None else [1001])

    def train_and_evaluate(
        self,
        pretrained: bool,
        seed: int,
        train_loader,
        test_loader,
        checkpoint_path: Path | None = None,
    ) -> tuple[EvaluationMetrics, nn.Module]:
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

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        total_samples = len(all_targets)
        correct_count = tn + tp
        error_count = fp + fn
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        f1 = float(f1_score(all_targets, all_preds, average="macro"))
        acc = float(accuracy_score(all_targets, all_preds))
        metrics = EvaluationMetrics(
            f1_macro=round(f1, 4),
            accuracy=round(acc, 4),
            total_samples=total_samples,
            correct_count=correct_count,
            error_count=error_count,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            sensitivity=round(sensitivity, 4),
            specificity=round(specificity, 4),
        )

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint salvo em: {checkpoint_path}")

        return metrics, model

    def get_run_filepath(self, seed: int) -> Path:
        return self.runs_dir / f"seed_{seed}.json"

    def is_seed_completed(self, seed: int) -> bool:
        return self.get_run_filepath(seed).exists()

    def load_seed_run(self, seed: int) -> PairedSimulationResult | None:
        path = self.get_run_filepath(seed)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PairedSimulationResult(**data)

    def run_single_seed(
        self,
        seed: int,
        test_loader,
        resume: bool = True,
        force_save_checkpoint: bool = False,
    ) -> PairedSimulationResult:
        """Executa ou recupera a simulação de uma única semente."""
        if resume and self.is_seed_completed(seed):
            saved = self.load_seed_run(seed)
            if saved is not None:
                logger.info(f"Semente {seed} recuperada do cache ({self.get_run_filepath(seed)}).")
                return saved

        train_loader = self.repo.get_train_loader(
            batch_size=self.batch_size, seed=seed, shuffle=True
        )

        save_cp = force_save_checkpoint or (seed in self.checkpoint_seeds)
        cp_transfer = self.checkpoints_dir / f"seed_{seed}_transfer.pt" if save_cp else None
        cp_scratch = self.checkpoints_dir / f"seed_{seed}_scratch.pt" if save_cp else None

        # Cenário 1: Transfer Learning (ImageNet)
        metrics_s1, _ = self.train_and_evaluate(
            pretrained=True,
            seed=seed,
            train_loader=train_loader,
            test_loader=test_loader,
            checkpoint_path=cp_transfer,
        )

        # Cenário 2: Scratch com Warmup + Cosine Annealing
        metrics_s2, _ = self.train_and_evaluate(
            pretrained=False,
            seed=seed,
            train_loader=train_loader,
            test_loader=test_loader,
            checkpoint_path=cp_scratch,
        )

        paired = PairedSimulationResult(
            seed=seed, scenario_1=metrics_s1, scenario_2=metrics_s2
        )

        # Persistência granular imediata
        run_file = self.get_run_filepath(seed)
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(paired.model_dump(), f, indent=2)

        return paired

    def run_all_simulations(
        self,
        resume: bool = True,
        save_all_checkpoints: bool = False,
    ) -> list[PairedSimulationResult]:
        """Executa as simulações com persistência incremental tolerante a falhas."""
        results: list[PairedSimulationResult] = []
        test_loader = self.repo.get_test_loader(batch_size=self.batch_size)

        logger.info(
            f"Iniciando {len(self.seeds)} simulações no dispositivo: {self.device} (resume={resume})"
        )

        for seed in tqdm(self.seeds, desc=f"Simulações Pareadas ({len(self.seeds)} sementes)"):
            paired = self.run_single_seed(
                seed=seed,
                test_loader=test_loader,
                resume=resume,
                force_save_checkpoint=save_all_checkpoints,
            )
            results.append(paired)

            # Atualização incremental do consolidado
            self.save_results(results, filepath=self.consolidated_path)

        return results

    def save_results(
        self,
        results: list[PairedSimulationResult],
        filepath: str | Path | None = None,
    ) -> None:
        path = Path(filepath or self.consolidated_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ordena por semente para consistência
        sorted_results = sorted(results, key=lambda r: r.seed)
        data = [r.model_dump() for r in sorted_results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def get_all_saved_runs(results_dir: str | Path = "results") -> list[PairedSimulationResult]:
        """Carrega todas as runs salvas no diretório results/runs/."""
        runs_dir = Path(results_dir) / "runs"
        if not runs_dir.exists():
            return []
        runs = []
        for file in sorted(runs_dir.glob("seed_*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    runs.append(PairedSimulationResult(**json.load(f)))
            except Exception as e:
                logger.warning(f"Erro ao ler run {file}: {e}")
        return sorted(runs, key=lambda r: r.seed)
