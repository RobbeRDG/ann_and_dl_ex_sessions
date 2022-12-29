from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from torchvision.datasets import Caltech101
from torchvision.transforms import transforms as tfm


class Baltech101(Caltech101):
    """
    Balanced version of the Caltech 101 dataset.
    See also <https://pytorch.org/vision/stable/generated/torchvision.datasets.Caltech101.html>.
    """
    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        seed: int = 42,
    ):
        super().__init__(
            root=root,
            target_type=target_type,
            transform=tfm.Compose([
                tfm.Lambda(lambda x: x.convert('RGB')),
                transform if transform is not None else tfm.Lambda(lambda x: x)
            ]),
            target_transform=target_transform,
            download=download,
        )

        np.random.seed(seed)

        y_uniq, first_idxs, counts = np.unique(self.y, return_index=True, return_counts=True)
        self.balanced_idxs = (
            first_idxs[:, None]
            + np.stack([
                np.random.randint(0, count, counts.min()) for count in counts
            ])
        ).flatten()

    def __len__(self):
        return len(self.balanced_idxs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        return super().__getitem__(self.balanced_idxs[index])