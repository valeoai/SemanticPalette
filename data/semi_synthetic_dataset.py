import torch
import torch.utils.data as data

from models.img_generator.models.img_model import ImgModel


class SemiSyntheticDataset(data.Dataset):
    def __init__(self, opt, dataset_size, batch_size, dataloader):
        self.img_model = ImgModel(opt["img_generator"], is_train=False)
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.loader_iter = iter(self.dataloader)
        self.latent_dim_img = opt["img_generator"].latent_dim
        self.batch_per_epoch = dataset_size // batch_size

    def next_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.dataloader)
            return next(self.loader_iter)

    def __len__(self):
        return self.batch_per_epoch

    def __getitem__(self, idx):
        with torch.no_grad():
            data = self.next_batch()
            data_img = self.img_model(data=data, mode='inference')
            data.update(data_img)
        return data
