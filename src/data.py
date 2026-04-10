from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 channel-wise statistics: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L32
_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    normalize = [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), *normalize] if augment else normalize
    )

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose(normalize))

    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
