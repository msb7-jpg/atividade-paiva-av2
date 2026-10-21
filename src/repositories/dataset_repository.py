from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}

# ImageNet statistics for ResNet-50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """Técnica A: Data Augmentation estocástico com transformações geométricas e fotométricas."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_test_transforms() -> transforms.Compose:
    """Transformação determinística para avaliação e teste imutável."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class XRayDataset(Dataset):
    """Dataset para Radiografias de Tórax (Chest X-Ray)."""

    def __init__(self, root_dir: str | Path, transform: transforms.Compose | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = self.root_dir / class_name
            if class_folder.exists():
                for img_path in sorted(class_folder.glob("*.jpg")):
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_raw_image(self, idx: int) -> tuple[Image.Image, int, Path]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label, img_path


class DatasetRepository:
    """Repositório de dados gerenciador de datasets e DataLoaders."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"

    def get_train_dataset(self, augment: bool = True) -> XRayDataset:
        transform = get_train_transforms() if augment else get_test_transforms()
        return XRayDataset(self.train_dir, transform=transform)

    def get_test_dataset(self) -> XRayDataset:
        return XRayDataset(self.test_dir, transform=get_test_transforms())

    def get_train_loader(
        self, batch_size: int = 64, seed: int = 1001, shuffle: bool = True
    ) -> DataLoader:
        dataset = self.get_train_dataset(augment=True)
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=4,
        )

    def get_test_loader(self, batch_size: int = 64) -> DataLoader:
        dataset = self.get_test_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

    def get_fixed_audit_samples(self) -> dict[str, tuple[Image.Image, Path]]:
        """Recupera instâncias fixas determinísticas para auditoria com Grad-CAM e LMM."""
        test_ds = self.get_test_dataset()
        samples: dict[str, tuple[Image.Image, Path]] = {}

        for i in range(len(test_ds)):
            raw_img, label, path = test_ds.get_raw_image(i)
            class_name = IDX_TO_CLASS[label]
            if class_name not in samples:
                samples[class_name] = (raw_img, path)
            if len(samples) == 2:
                break
        return samples
