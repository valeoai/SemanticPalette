import torch
import torch.utils.data as data

from models.seg_generator.models.seg_model import SegModel as SegModelGen
from models.seg_completor.models.seg_model import SegModel as SegModelCom
from models.img_generator.models.img_model import ImgModel
from tools.cond_sampler import CondSampler
from tools.cond_estimator import CondEstimator


class SyntheticDataset(data.Dataset):
    def __init__(self, opt, dataset_size, batch_size, source_dataloader):
        if opt["seg_generator"].seg_type == "generator":
            self.seg_model = SegModelGen(opt["seg_generator"], is_train=False)
        elif opt["seg_generator"].seg_type == "completor":
            self.seg_model = SegModelCom(opt["seg_generator"], is_train=False)
            self.seg_dataloader = source_dataloader
            self.seg_loader_iter = iter(self.seg_dataloader)
        else:
            raise ValueError
        self.img_model = ImgModel(opt["img_generator"], is_train=False)
        self.batch_size = batch_size
        self.cond_sampler = CondSampler(opt)
        self.duo_cond = opt["segmentor"].duo_cond
        if self.duo_cond:
            self.cond_estimator_duo = CondEstimator(opt["extra_dataset"])
        self.latent_dim_seg = opt["seg_generator"].latent_dim
        self.latent_dim_img = opt["img_generator"].latent_dim
        self.batch_per_epoch = dataset_size // batch_size
        self.opt = opt

    def __len__(self):
        return self.batch_per_epoch

    def next_seg_batch(self):
        try:
            return next(self.seg_loader_iter)
        except StopIteration:
            self.seg_loader_iter = iter(self.seg_dataloader)
            return next(self.seg_loader_iter)

    def __getitem__(self, idx):
        with torch.no_grad():
            data = {"z_seg": torch.randn(self.batch_size, self.latent_dim_seg)}
            if self.opt["seg_generator"].seg_type == "completor":
                data["sem_seg"] = self.next_seg_batch()["sem_seg"]
            if self.duo_cond:
                cond_real = self.cond_sampler.sample_batch(self.batch_size // 2)
                tgt_sem_cond, tgt_ins_cond = self.cond_estimator_duo.sample(n_samples=self.batch_size // 2)
                cond = {"sem_cond": torch.cat([cond_real["sem_cond"], tgt_sem_cond], dim=0),
                        "ins_cond": torch.cat([cond_real["ins_cond"], tgt_ins_cond], dim=0)}
            else:
                cond = self.cond_sampler.sample_batch(self.batch_size)
            data.update(cond)
            data_seg = self.seg_model(data=data, mode='inference', hard=True)
            data_seg = self.conv_seg(data_seg)
            data_img = self.img_model(data=data_seg, mode='inference')
            data.update(data_seg)
            data.update(data_img)
        return data

    def set_batch_delta(self, delta):
        delta = self.conv_delta(delta)
        if self.duo_cond:
            delta["batch_delta"] = delta["batch_delta"][:self.batch_size // 2]
        self.cond_sampler.set_batch_delta(delta)

    def update_sampler(self, logger, log, global_iteration, save=False):
        self.cond_sampler.update(logger, log, global_iteration, save)

    def conv_seg(self, data_seg):
        data_seg["sem_seg"] = self.conv(data_seg["sem_seg"], mode="seg_to_img", dim=1)
        return data_seg

    def conv_delta(self, delta):
        delta["class_delta"] = self.conv(delta["class_delta"], mode="img_to_seg", dim=0)
        delta["class_num"] = self.conv(delta["class_num"], mode="img_to_seg", dim=0)
        delta["class_surface"] = self.conv(delta["class_surface"], mode="img_to_seg", dim=0)
        return delta

    def conv(self, tensor, mode, dim):
        if self.opt["seg_generator"].sem_conv is not None:
            conv_shape = list(tensor.shape)
            conv_num_semantics = self.opt["img_generator"].num_semantics if mode == "seg_to_img" else self.opt["seg_generator"].num_semantics
            conv_shape[dim] = conv_num_semantics
            conv_tensor = torch.zeros(conv_shape).to(tensor.get_device())
            for i, j in self.opt["seg_generator"].sem_conv.items():
                if mode == "seg_to_img":
                    i, j = i, j
                elif mode == "img_to_seg":
                    i, j = j, i
                else:
                    raise ValueError
                if dim == 0:
                    conv_tensor[j] = tensor[i]
                elif dim == 1:
                    conv_tensor[:, j] = tensor[:, i]
            return conv_tensor
        return tensor



