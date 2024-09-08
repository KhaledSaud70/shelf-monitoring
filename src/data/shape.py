import os
import tempfile
import zipfile
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SHAPEDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None,
    ):
        self.root = root
        self.transform = transform

        self.temp_dir = None
        self.root_to_walk = self._prepare_root()

        self._classes, self.class_to_idx = self._find_classes(class_to_idx)
        self.samples = self._make_dataset()

    def _prepare_root(self):
        if os.path.isfile(self.root) and self.root.lower().endswith(".zip"):
            self.temp_dir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(self.root, "r") as zip_ref:
                zip_ref.extractall(self.temp_dir.name)
            return self.temp_dir.name
        return self.root

    def _find_classes(self, class_to_idx: Optional[dict] = None) -> Tuple[list, dict]:
        if class_to_idx is not None:
            return list(class_to_idx.keys()), class_to_idx

        classes = set()
        for category in os.listdir(self.root_to_walk):
            category_path = os.path.join(self.root_to_walk, category)
            if os.path.isdir(category_path):
                for class_name in os.listdir(category_path):
                    classes.add(class_name)
        classes = sorted(list(classes))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> list:
        samples = []
        for root, _, filenames in os.walk(self.root_to_walk):
            if len(filenames) > 0:
                category, class_name = os.path.split(os.path.dirname(root))
                if class_name in self.class_to_idx:
                    for img_name in filenames:
                        img_path = os.path.join(root, img_name)
                        if self._is_valid_file(img_path):
                            if self.root_to_walk != self.root:
                                relative_path = os.path.relpath(img_path, self.root_to_walk)
                                img_path = os.path.join(self.root, relative_path)
                            item = (img_path, self.class_to_idx[class_name])
                            samples.append(item)
        return samples

    def _is_valid_file(self, filename: str) -> bool:
        return filename.lower().endswith((".png", ".jpg", ".jpeg"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    @property
    def classes(self) -> list:
        return self._classes

    def __del__(self):
        if self.temp_dir:
            self.temp_dir.cleanup()


if __name__ == "__main__":
    data_dir = "/home/khaled/workspace/projects/shelf-monitoring/src/data/training_set.zip"
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = SHAPEDataset(data_dir, transform)

    # Use RandomSampler for shuffling during training
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Example: Iterate through the train loader
    for images, labels, categories in train_loader:
        print(images.shape, labels, categories)
        break
