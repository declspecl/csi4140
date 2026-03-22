from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    use_augmentation: bool = True,
) -> tuple[DataLoader, DataLoader]:
    # CIFAR-10 channel-wise statistics: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L32
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                *normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(normalize)

    test_transform = transforms.Compose(normalize)

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
