################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2020                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################
from pathlib import Path
from typing import Union, Any, Optional

from torchvision import transforms, datasets

from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets import TinyImagenet
from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset


_default_train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

_default_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


def Nette(
    n_experiences=5,
    *,
    return_task_id=False,
    seed=0,
    fixed_class_order=None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None
):
    train_set, test_set = _get_tiny_imagenet_dataset(dataset_root)

    return nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def _get_tiny_imagenet_dataset(dataset_root):   
    try:
        train_set = datasets.Imagenette('~/data', split='train', download=True, size='full')
        test_set = datasets.Imagenette('~/data', split='val', download=True, size='full')
    except:
        train_set = datasets.Imagenette('~/data', split='train', download=False, size='full')
        test_set = datasets.Imagenette('~/data', split='val', download=False, size='full')

    data = []
    targets = []
    
    for img, label in train_set:
        data.append(img)
        targets.append(label)

    train_set.data = data
    train_set.targets = targets
    
    data = []
    targets = []
    
    for img, label in test_set:
        data.append(img)
        targets.append(label)

    test_set.data = data
    test_set.targets = targets

    train_set = as_classification_dataset(train_set)
    test_set = as_classification_dataset(test_set)

    return train_set, test_set


if __name__ == "__main__":
    import sys

    benchmark_instance = Nette()
    check_vision_benchmark(benchmark_instance)
    sys.exit(0)

__all__ = ["Nette"]
