import importlib
import torch.utils.data
import numpy as np

from data.base_dataset import BaseDataset
from data.synthetic_dataset import SyntheticDataset
from data.semi_synthetic_dataset import SemiSyntheticDataset

def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported. 
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
            
    if dataset is None:
        raise ValueError(f"In {dataset_filename}.py, there should be a subclass of BaseDataset "
                         f"with class name that matches {target_dataset_name} in lowercase.")

    return dataset


def create_dataset(opt, load_seg=False, load_img=False, phase='train'):
    dataset = find_dataset_using_name(opt.dataset)
    instance = dataset(load_seg, load_img, opt, phase=phase)
    print(f"Creation of dataset [{type(instance).__name__}-{phase}] of size {len(instance)}")
    return instance


def create_synthetic_dataset(opt, batch_size, phase='train'):
    source_dataset = find_dataset_using_name(opt['base'].dataset)
    instance = source_dataset(True, False, opt=opt['seg_generator'], phase=phase)
    dataset_size = len(instance)
    print(f"Creation of synthetic dataset [{opt['base'].dataset}-{phase}] of size {dataset_size}")
    if opt["seg_generator"].seg_type == "completor":
        source_dataloader = create_dataloader(instance, batch_size, opt['base'].num_workers, phase == 'train')
    else:
        source_dataloader = None
    return SyntheticDataset(opt, dataset_size, batch_size, source_dataloader)


def create_semi_synthetic_dataset(opt, batch_size, phase='train'):
    source_dataset = find_dataset_using_name(opt['base'].dataset)
    instance = source_dataset(True, False, opt=opt['base'], phase=phase)
    dataset_size = len(instance)
    source_dataloader = create_dataloader(instance, batch_size, opt['base'].num_workers, phase=='train')
    print(f"Creation of semi synthetic dataset [{opt['base'].dataset}-{phase}] of size {dataset_size}")
    return SemiSyntheticDataset(opt, dataset_size, batch_size, source_dataloader)


def create_dataloader(dataset, batch_size, num_workers, is_train):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
        worker_init_fn=lambda _: np.random.seed()
    )
    return dataloader
