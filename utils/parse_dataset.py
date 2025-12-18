import torchvision.datasets as datasets
from torchvision import transforms


def get_transforms(transform_config):
    transform_list = []
    for step in transform_config.steps:
        transform_type = getattr(transforms, step.type)
        if 'params' in step:
            transform_list.append(transform_type(**step.params))
        else:
            transform_list.append(transform_type())
    return transforms.Compose(transform_list)


def initialize_dataset(config):
    # Check if the dataset exists in torchvision.datasets

    dataset_class = getattr(datasets, config.dataset)
    root = getattr(config, "data_dir", "data")
    train_transform = get_transforms(config.transform_train)
    test_transform = get_transforms(config.transform_test)
    train_dataset = dataset_class(
        root=root,
        train=True,
        transform=train_transform,
        download=True  # Download the dataset if not already present
    )
    test_dataset = dataset_class(
        root=root,
        train=False,
        transform=test_transform,
        download=True  # Download the dataset if not already present
    )
    return train_dataset, test_dataset
