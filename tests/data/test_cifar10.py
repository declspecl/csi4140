from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.data import get_dataloaders


def _fake_cifar10(root, train, download, transform):
    n = 200 if train else 100
    images = torch.rand(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    return TensorDataset(images, labels)


@pytest.fixture
def mock_cifar10():
    with patch("src.data.datasets.CIFAR10", side_effect=_fake_cifar10) as mock:
        yield mock


class TestGetDataloaders:
    def test_returns_two_dataloaders(self, mock_cifar10):
        train_loader, test_loader = get_dataloaders()
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_default_batch_size(self, mock_cifar10):
        train_loader, test_loader = get_dataloaders()
        assert train_loader.batch_size == 128
        assert test_loader.batch_size == 128

    def test_custom_batch_size(self, mock_cifar10):
        train_loader, test_loader = get_dataloaders(batch_size=64)
        assert train_loader.batch_size == 64
        assert test_loader.batch_size == 64

    def test_batch_shapes(self, mock_cifar10):
        train_loader, test_loader = get_dataloaders(batch_size=32)
        images, labels = next(iter(train_loader))
        assert images.shape == (32, 3, 32, 32)
        assert labels.shape == (32,)

    def test_label_dtype(self, mock_cifar10):
        train_loader, _ = get_dataloaders(batch_size=16)
        _, labels = next(iter(train_loader))
        assert labels.dtype == torch.int64

    def test_train_shuffle_enabled(self, mock_cifar10):
        train_loader, _ = get_dataloaders()
        assert type(train_loader.sampler).__name__ == "RandomSampler"

    def test_test_shuffle_disabled(self, mock_cifar10):
        _, test_loader = get_dataloaders()
        assert type(test_loader.sampler).__name__ == "SequentialSampler"

    def test_dataset_download_true(self, mock_cifar10):
        get_dataloaders(data_dir="/tmp/test_data")
        for call in mock_cifar10.call_args_list:
            assert call.kwargs["download"] is True

    def test_custom_data_dir(self, mock_cifar10):
        get_dataloaders(data_dir="/tmp/custom")
        for call in mock_cifar10.call_args_list:
            assert call.kwargs["root"] == "/tmp/custom"

    def test_train_and_test_splits(self, mock_cifar10):
        get_dataloaders()
        calls = mock_cifar10.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["train"] is True
        assert calls[1].kwargs["train"] is False

    def test_custom_num_workers(self, mock_cifar10):
        train_loader, test_loader = get_dataloaders(num_workers=4)
        assert train_loader.num_workers == 4
        assert test_loader.num_workers == 4


class TestAugmentation:
    def test_augmentation_enabled_has_four_transforms(self, mock_cifar10):
        get_dataloaders(augment=True)
        train_transform = mock_cifar10.call_args_list[0].kwargs["transform"]
        assert len(train_transform.transforms) == 4
        assert isinstance(train_transform.transforms[0], transforms.RandomHorizontalFlip)
        assert isinstance(train_transform.transforms[1], transforms.RandomCrop)
        assert isinstance(train_transform.transforms[2], transforms.ToTensor)
        assert isinstance(train_transform.transforms[3], transforms.Normalize)

    def test_augmentation_disabled_has_two_transforms(self, mock_cifar10):
        get_dataloaders(augment=False)
        train_transform = mock_cifar10.call_args_list[0].kwargs["transform"]
        assert len(train_transform.transforms) == 2
        assert isinstance(train_transform.transforms[0], transforms.ToTensor)
        assert isinstance(train_transform.transforms[1], transforms.Normalize)

    def test_test_transform_never_augmented(self, mock_cifar10):
        get_dataloaders(augment=True)
        test_transform = mock_cifar10.call_args_list[1].kwargs["transform"]
        assert len(test_transform.transforms) == 2
        assert isinstance(test_transform.transforms[0], transforms.ToTensor)
        assert isinstance(test_transform.transforms[1], transforms.Normalize)

    def test_random_crop_padding(self, mock_cifar10):
        get_dataloaders(augment=True)
        crop = mock_cifar10.call_args_list[0].kwargs["transform"].transforms[1]
        assert crop.size == (32, 32)
        assert crop.padding == 4


class TestNormalization:
    def test_normalize_mean(self, mock_cifar10):
        get_dataloaders()
        normalize = mock_cifar10.call_args_list[0].kwargs["transform"].transforms[-1]
        assert tuple(normalize.mean) == (0.4914, 0.4822, 0.4465)

    def test_normalize_std(self, mock_cifar10):
        get_dataloaders()
        normalize = mock_cifar10.call_args_list[0].kwargs["transform"].transforms[-1]
        assert tuple(normalize.std) == (0.2470, 0.2435, 0.2616)

    def test_train_and_test_use_same_normalization(self, mock_cifar10):
        get_dataloaders()
        train_norm = mock_cifar10.call_args_list[0].kwargs["transform"].transforms[-1]
        test_norm = mock_cifar10.call_args_list[1].kwargs["transform"].transforms[-1]
        assert tuple(train_norm.mean) == tuple(test_norm.mean)
        assert tuple(train_norm.std) == tuple(test_norm.std)
