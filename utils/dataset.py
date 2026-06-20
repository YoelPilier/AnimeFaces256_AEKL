import datasets
import torch


def load_dataset_from_local_or_hub(dataset_name, base_dir, split="train"):
    """
    Load a dataset from the Hugging Face Hub or from a local directory.
    """
    assert base_dir is not None, "Base directory must be provided"

    try:
        dataset = datasets.load_from_disk(base_dir / dataset_name.split("/")[1])
        print(f"Loaded dataset from local directory: {dataset_name.split('/')[1]}")
    except Exception as e:
        print(f"Failed to load dataset from local directory: {e}")
        print(f"Loading dataset from Hugging Face Hub: {dataset_name}")
        dataset = datasets.load_dataset(dataset_name, split=split)
        print(f"Saving dataset to local directory: {dataset_name.split('/')[1]}")
        dataset.save_to_disk(base_dir / dataset_name.split("/")[1])
    return dataset


def create_train_test_split(
    *datasets_list,
    seed: int = 42,
    test_percentage: float = 0.1,
):
    dataset = datasets.concatenate_datasets(list(datasets_list))
    dataset = dataset.shuffle(seed=seed)

    split = dataset.train_test_split(
        test_size=test_percentage,
        seed=seed,
    )

    return split["train"], split["test"]


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


def to_image_range(tensor, de_normalize):
    return torch.clamp(de_normalize(tensor.float()), 0, 1)
