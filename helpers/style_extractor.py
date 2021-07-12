from tqdm import tqdm
from glob import glob
import numpy as np

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urbangan_dir = os.path.dirname(current_dir)
sys.path.insert(0, urbangan_dir)

from tools.options import Options
from tools.engine import Engine
from data import create_dataset
from models.img_style_generator.models.img_model import ImgModel


class StyleExtractor:
    def __init__(self, opt):
        self.opt = opt

    def compute_style(self, data):
        self.img_model(data, mode='inference')

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            self.dataset = create_dataset(self.opt, load_seg=True, load_img=True)
            self.dataloader, self.datasampler = engine.create_dataloader(self.dataset, self.opt.batch_size, self.opt.num_workers, False)
            is_main = self.opt.local_rank == 0
            self.img_model_on_one_gpu = ImgModel(self.opt, is_train=False, is_main=is_main)
            self.img_model = engine.data_parallel(self.img_model_on_one_gpu)
            self.img_model.eval()
            self.img_model.netG.set_status("test")

            for i, data_i in tqdm(enumerate(self.dataloader), desc="Computing individual styles"):
                data_i["obj_dic"] = [f"img_{int(i * self.opt.batch_size + j)}.jpg" for j in range(self.opt.batch_size)]
                self.compute_style(data_i)
                if i * self.opt.batch_size > self.opt.niter:
                    break

            layer = 'ACE.npy'
            for cat_i in tqdm(range(self.opt.num_semantics), desc="Computing mean styles"):
                tmp_list = glob(os.path.join(self.opt.log_path, 'style_codes/*', str(cat_i), layer))
                style_list = []

                for k in tmp_list:
                    style_list.append(np.load(k))

                if len(style_list) > 0:
                    print(f"Found {len(style_list)} references for style {cat_i}")
                    result = np.array(style_list).mean(0)
                    save_folder = os.path.join(self.opt.log_path, 'mean_style_code/mean', str(cat_i))
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_name = os.path.join(save_folder, layer)
                    np.save(save_name, result)

            print('Extraction was successfully finished.')


if __name__ == "__main__":
    opt = Options().parse(load_img_generator=True, save=True)
    StyleExtractor(opt["img_generator"]).run()