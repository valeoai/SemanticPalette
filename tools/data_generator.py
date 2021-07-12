import torch
import os
import numpy as np

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urbangan_dir = os.path.dirname(current_dir)
sys.path.insert(0, urbangan_dir)

from tools.options import Options
from tools.cond_sampler import CondSampler
from tools.utils import save_image, dict_to_cuda, create_colormap, color_transfer
from models.seg_generator.models.seg_model import SegModel
from models.img_generator.models.img_model import ImgModel

class DataGenerator:
    def __init__(self, opt):
        self.seg_model = SegModel(opt["seg_generator"], is_train=False)
        self.img_model = ImgModel(opt["img_generator"], is_train=False)
        self.batch_size = opt["base"].batch_size
        self.cond_sampler = CondSampler(opt)
        self.latent_dim_seg = opt["seg_generator"].latent_dim
        self.latent_dim_img = opt["img_generator"].latent_dim

    def next_batch(self, global_iteration=0, dim_ind=0, log=False):
        with torch.no_grad():
            z_seg = torch.randn(self.batch_size, self.latent_dim_seg).cuda()
            # z_img = torch.randn(self.batch_size, self.latent_dim_img)
            cond = dict_to_cuda(self.cond_sampler.sample(self.batch_size))
            seg = self.seg_model(global_iteration, dim_ind, log=log, real_cond=cond, z=z_seg, mode='inference', hard=True)
            data = self.seg_to_data(seg)
            img = self.img_model(data=data, one_hot=True, mode='inference')
        return seg["sem_seg"], img

    def seg_to_data(self, seg):
        data = seg
        data["img"] = torch.Tensor([])
        data["seg"] = torch.Tensor([])
        return data

if __name__ == "__main__":
    opt = Options().parse(load_seg_generator=True, load_img_generator=True, save=False)
    data_generator = DataGenerator(opt)
    colormap = create_colormap(opt["base"])
    seg, img = data_generator.next_batch()

    img = (0.5 * (img[0].permute(1, 2, 0).cpu().numpy() + 1) * 255).astype(np.uint8)
    seg = (0.5 * (color_transfer(seg.max(dim=1, keepdim=True)[1], colormap)[0].permute(1, 2, 0).cpu().numpy() + 1) * 255).astype(np.uint8)

    img_path = os.path.join(opt["base"].save_path, "img.png")
    seg_path = os.path.join(opt["base"].save_path, "seg.png")

    save_image(img, img_path)
    save_image(seg, seg_path)

