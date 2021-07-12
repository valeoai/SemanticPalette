import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import imageio
import numpy as np
import pandas as pd
from copy import deepcopy
import random
from glob import glob
import math

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urbangan_dir = os.path.dirname(current_dir)
sys.path.insert(0, urbangan_dir)

from tools.options import Options
from tools.engine import Engine
from tools.logger import Logger
from data import create_dataset
from tools.utils import create_colormap, color_transfer, color_spread, save_image
from tools.cond_sampler import CondSampler
from models.seg_img_model import SegImgModel
from models.seg_generator.modules.instance_refiner import InstanceRefiner


class Visualizer:
    def __init__(self, opt):
        self.opt = opt["base"]
        self.seg_opt = opt["seg_generator"]
        self.img_opt = opt["img_generator"]
        self.cond_sampler = CondSampler(opt)
        self.batch_size = self.opt.batch_size
        self.vis_steps = self.opt.vis_steps
        self.ins_refiner = InstanceRefiner(self.opt)

    def save_full_res(self, savefile, real_cond, fake_cond, imgs, real_ins=None, fake_ins=None):
        save_dir = os.path.basename(savefile).split('.')[0]
        save_folder = os.path.join(self.opt.log_path, save_dir)
        print("Saving imgs at full res in", save_folder)
        os.mkdir(save_folder)

        # save img to png
        for i, img_list in enumerate(imgs):
            for j, img in enumerate(img_list):
                img_path = os.path.join(save_folder, f"img_{i}_{j}.png")
                np_img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
                save_image(np_img, img_path)

        # save prop to csv
        csv_file = os.path.join(save_folder, "prop.csv")
        excel_file = os.path.join(save_folder, "prop.xlsx")
        columns = ["img_id", "prop"] + self.opt.semantic_labels
        if real_ins is not None:
            columns += ["ins_" + self.opt.semantic_labels[k] for k in self.opt.things_idx]
            real_ins = real_ins.numpy()
            fake_ins = fake_ins.numpy()
        df = pd.DataFrame(columns=columns)
        real_cond = real_cond.numpy()
        fake_cond = fake_cond.numpy()
        for i in range(len(real_cond)):
            real_raw = {"img_id": str(i), "prop": "target"}
            gen_raw = {"img_id": str(i), "prop": "generated"}
            for j, label in enumerate(self.opt.semantic_labels):
                real_raw[label] = real_cond[i][j]
                gen_raw[label] = fake_cond[i][j]
            if real_ins is not None:
                for j, idx in enumerate(self.opt.things_idx):
                    label = "ins_" + self.opt.semantic_labels[idx]
                    real_raw[label] = real_ins[i][j]
                    gen_raw[label] = fake_ins[i][j]
            df = df.append(real_raw, ignore_index=True)
            df = df.append(gen_raw, ignore_index=True)
        df.to_csv(csv_file, sep=",", index=False)
        df.to_excel(excel_file)

    def fixedsem_manipulation(self, savefile):
        data = self.next_batch()
        data["z_seg"] = torch.randn(self.vis_steps, self.seg_opt.latent_dim)
        data["sem_cond"] = data["sem_cond"][0].repeat(self.vis_steps, 1)
        fake_data = []
        batch_num = self.vis_steps // self.batch_size
        for i in range(batch_num):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.vis_steps)
            batch_data = {k: v[start:end] if v.size(0) > 0 else v for k, v in data.items()}
            fake_data.append(self.seg_img_model(data=batch_data, mode='inference', log=True, as_list=False))
        fake_data = {k: torch.cat([fake[k] for fake in fake_data], dim=0) for k in fake_data[0].keys()}

        fake_sem_seg = fake_data["sem_seg"]
        fake_sem_cond = torch.mean(fake_sem_seg, dim=(2, 3)).cpu()
        fake_pred = torch.argmax(fake_sem_seg, dim=1, keepdim=True).cpu()
        fake_sem_img = (color_transfer(fake_pred, self.colormap) + 1) / 2
        fake_img = (fake_data["img"].cpu() + 1) / 2

        real_sem_seg = data["sem_seg"]
        real_sem_cond = data["sem_cond"].cpu()
        real_pred = torch.argmax(real_sem_seg, dim=1, keepdim=True).cpu()
        real_sem_img = (color_transfer(real_pred, self.colormap) + 1) / 2
        real_img = (data["img"].cpu() + 1) / 2

        if self.opt.save_full_res:
            self.save_full_res(savefile, real_sem_cond, fake_sem_cond, [real_img,
                                                                        real_sem_img,
                                                                        fake_img,
                                                                        fake_sem_img])

    def scrop_manipulation(self, savefile):
        data = self.next_batch()
        data["z_seg"] = torch.randn(self.vis_steps, self.seg_opt.latent_dim)
        fake_data = []
        batch_num = self.vis_steps // self.batch_size
        for i in range(batch_num):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.vis_steps)
            batch_data = {k: v[start:end] if v.size(0) > 0 else v for k, v in data.items()}
            fake_data.append(self.seg_img_model(data=batch_data, mode='inference', log=True, as_list=False))
        fake_data = {k: torch.cat([fake[k] for fake in fake_data], dim=0) for k in fake_data[0].keys()}

        fake_sem_seg = fake_data["sem_seg"]
        fake_sem_cond = torch.mean(fake_sem_seg, dim=(2, 3)).cpu()
        fake_pred = torch.argmax(fake_sem_seg, dim=1, keepdim=True).cpu()
        fake_raw = self.process_raw_sem(fake_data)
        fake_sem_img = (color_transfer(fake_pred, self.colormap) + 1) / 2
        fake_img = (fake_data["img"].cpu() + 1) / 2
        fake_raw_img = (color_transfer(fake_raw, self.colormap) + 1) / 2

        real_sem_seg = data["sem_seg"]
        real_sem_cond = torch.mean(real_sem_seg, dim=(2, 3)).cpu()
        real_pred = torch.argmax(real_sem_seg, dim=1, keepdim=True).cpu()
        real_raw = fake_data["real_cropped"].cpu()
        real_raw_pred = torch.argmax(real_raw, dim=1, keepdim=True)
        real_raw_pred[torch.max(real_raw, dim=1, keepdim=True)[0] == 1. / self.opt.num_semantics] = -1
        real_sem_img = (color_transfer(real_pred, self.colormap) + 1) / 2
        real_img = (data["img"].cpu() + 1) / 2
        real_raw_img = (color_transfer(real_raw_pred, self.colormap) + 1) / 2

        if self.opt.save_full_res:
            self.save_full_res(savefile, real_sem_cond, fake_sem_cond, [real_img,
                                                                        real_sem_img,
                                                                        real_raw_img,
                                                                        fake_img,
                                                                        fake_sem_img,
                                                                        fake_raw_img])


    def latent_interpolation(self, savefile):
        z0 = torch.randn(self.seg_opt.latent_dim)
        z1 = torch.randn(self.seg_opt.latent_dim)
        z = [z0 * i / (self.batch_size - 1) + z1 * (1 - i / (self.batch_size - 1)) for i in range(self.batch_size)]
        data = {"z_seg": torch.stack(z)}
        c = self.cond_sampler.sample_batch(1)
        ins_cond = torch.cat([c["ins_cond"]] * self.batch_size, dim=0)
        sem_cond = torch.cat([c["sem_cond"]] * self.batch_size, dim=0)
        cond = {"ins_cond": ins_cond, "sem_cond": sem_cond}
        data.update(cond)
        fake_data = self.seg_img_model(data=data, mode='inference', log=True, as_list=False)
        if self.opt.vis_ins:
            self.plot_ins_interpolation(fake_data, ins_cond, sem_cond, savefile)
        else:
            self.plot_sem_interpolation(fake_data, sem_cond, savefile)

    def cond_interpolation(self, savefile):
        z = torch.randn(self.seg_opt.latent_dim)
        data = {"z_seg": torch.stack([z] * self.batch_size)}
        c0 = self.cond_sampler.sample_batch(1)
        ins0 = c0["ins_cond"]
        sem0 = c0["sem_cond"]
        c1 = self.cond_sampler.sample_batch(1)
        ins1 = c1["ins_cond"]
        sem1 = c1["sem_cond"]
        ins = [ins0 * i / (self.batch_size - 1) + ins1 * (1 - i / (self.batch_size - 1)) for i in range(self.batch_size)]
        sem = [sem0 * i / (self.batch_size - 1) + sem1 * (1 - i / (self.batch_size - 1)) for i in range(self.batch_size)]
        ins_cond = torch.cat(ins, dim=0)
        sem_cond = torch.cat(sem, dim=0)
        cond = {"ins_cond": ins_cond, "sem_cond": sem_cond}
        data.update(cond)
        fake_data = self.seg_img_model(data=data, mode='inference', log=True, as_list=False)
        if self.opt.vis_ins:
            self.plot_ins_interpolation(fake_data, ins_cond, sem_cond, savefile)
        else:
            self.plot_sem_interpolation(fake_data, sem_cond, savefile)

    def ins_manipulation(self, idx, sem_min, sem_max, ins_min, ins_max, savefile):
        z = torch.randn(self.seg_opt.latent_dim)
        data = {"z_seg": torch.stack([z] * self.vis_steps)}
        c0 = self.cond_sampler.sample_batch(1)
        ins0 = c0["ins_cond"]
        sem0 = c0["sem_cond"]
        sem0[0, idx] = sem_min
        sem0 /= torch.sum(sem0)
        sem1 = sem0.clone()
        sem1[0, idx] = sem_max
        sem1 /= torch.sum(sem1)
        thing_idx = self.opt.things_idx.index(idx)
        ins0[:, thing_idx] = ins_min
        ins1 = ins0.clone()
        ins1[:, thing_idx] = ins_max
        ins = [ins0 * i / (self.vis_steps - 1) + ins1 * (1 - i / (self.vis_steps - 1)) for i in range(self.vis_steps)]
        sem = [sem0 * i / (self.vis_steps - 1) + sem1 * (1 - i / (self.vis_steps - 1)) for i in range(self.vis_steps)]
        ins_cond = torch.cat(ins, dim=0)
        sem_cond = torch.cat(sem, dim=0)
        cond = {"ins_cond": ins_cond, "sem_cond": sem_cond}
        data.update(cond)

        fake_data = []
        batch_num = math.ceil(1. * self.vis_steps / self.batch_size)
        for i in range(batch_num):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.vis_steps)
            batch_data = {k: v[start:end] if v.size(0) > 0 else v for k, v in data.items()}
            fake_data.append(self.seg_img_model(data=batch_data, mode='inference', log=True, as_list=False))
        fake_data = {k: torch.cat([fake[k] for fake in fake_data], dim=0) for k in fake_data[0].keys()}
        self.plot_ins_interpolation(fake_data, ins_cond, sem_cond, savefile)

    def get_style(self):
        if self.opt.vis_random_style:
            return self.get_random_style()
        else:
            return self.get_mean_style()

    def get_random_style(self):
        obj_dic = {}
        for i in range(self.opt.num_semantics):
            if i == 5: # right and left eyes should have same style
                pass
            style_code_path = random.choice(self.style_codes[i])
            style = np.load(style_code_path)
            obj_dic[str(i)] = {'ACE': torch.tensor(style).cuda()}
            if i == 4: # right and left eyes should have same style
                style = np.load(style_code_path.replace("/4/", "/5/"))
                obj_dic[str(i + 1)] = {'ACE': torch.tensor(style).cuda()}

        return obj_dic

    def get_mean_style(self):
        obj_dic = {}
        for i in range(self.opt.num_semantics):
            folder_path = os.path.join(self.opt.extraction_path, "mean_style_code", "mean", str(i))
            style_code_path = os.path.join(folder_path, 'ACE.npy')
            style = np.load(style_code_path)
            obj_dic[str(i)] = {'ACE': torch.tensor(style).cuda()}
        return obj_dic

    def face_manipulation(self, savefile, bg_to_white=False, mode=None):
        data = self.next_batch()
        img = data["img"]
        sem_seg = data["sem_seg"]
        force_mean_idx = []

        if bg_to_white:
            bg_mask = sem_seg[:, 0] == 1
            img.permute(0,2,3,1)[bg_mask] = 1.

        data["img"] = img.repeat(self.vis_steps, 1, 1, 1)
        index = sem_seg.max(1, keepdim=True)[1]
        sem_seg = torch.zeros_like(sem_seg).scatter_(1, index, 1.0)
        sem0 = torch.mean(sem_seg.float(), dim=(2, 3))
        sem1 = data["sem_cond"]

        if mode == "earrings":
            if sem0[0, 15] != 0: # if already has earrings, sample new image
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            if sem0[0, 13] < 0.3: # hair is too short
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 15] = 0.03 # earrings
            sem1[0, 13] -= 0.03 # hair
            force_mean_idx = [15]

        if mode == "skin":
            if sem0[0, 13] > 0.08: # too much hair
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 1] += 0.06 # skin
            sem1[0, 0] -= 0.06  # bg

        if mode == "nose":
            if sem0[0, 13] > 0.08: # too much hair
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 1] -= 0.05 # skin
            sem1[0, 2] += 0.05  # nose

        if mode == "hat":
            if sem0[0, 14] != 0: # if already has hat, sample new image
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            if sem0[0, 13] < 0.3: # hair is too short
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 14] = 0.1 # hat
            sem1[0, 13] -= 0.07 # hair
            sem1[0, 0] -= 0.03 # bg
            force_mean_idx = [14]

        if mode == "glasses":
            if sem0[0, 3] != 0: # if already has glasses, sample new image
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 3] = 0.06 # glasses
            sem1[0, 1] -= 0.06 # skin
            force_mean_idx = [3]

        if mode == "openeyes":
            if sem0[0, 4] != 0 or sem0[0, 5] != 0: # if already has eyes open, sample new image
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            if sem0[0, 3] != 0: # if has glasses, sample new image
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 4] = 0.01 # l_eye
            sem1[0, 5] = 0.01  # r_eye
            sem1[0, 1] -= 0.02 # skin
            force_mean_idx = [4, 5]

        if mode == "unbald":
            if sem0[0, 13] != 0: # if already has hair, sample new image
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            if sem0[0, 14] != 0: # if has hat, hat might cover hair, sample new image
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 13] = 0.2 # hair
            sem1[0, 1] -= 0.15 # skin
            sem1[0, 0] -= 0.05 # bg
            force_mean_idx = [13]

        if mode == "bald":
            if sem0[0, 13] > 0.15: # too much hair
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            if sem0[0, 13] < 0.05: # not enough hair
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 13] = 0 # hair
            sem1[0, 1] += 3 * sem0[0, 13] / 4 # skin
            sem1[0, 0] += sem0[0, 13] / 4 # bg

        if mode == "teeth":
            if sem0[0, 10] != 0: # if already has teeth, sample new image
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 10] = 0.02 # mouth
            sem1[0, 1] -= 0.02 # skin
            if self.addition_mode:
                sem1[0, 11] += sem0[0, 11]  # u_lip
                sem1[0, 12] += sem0[0, 12]  # l_lip
            force_mean_idx = [10]

        if mode == "eyebrows":
            if sem0[0, 6] < 0.008: # not enough eyebrows
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 6] = 0.001 # l_brow
            sem1[0, 7] = 0.001 # r_brow
            sem1[0, 1] += sem0[0, 6] + sem0[0, 7] - 0.002 # skin

        if mode == "morebrows":
            if sem0[0, 6] > 0.001: # too much eyebrows
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 6] = 0.01 # l_brow
            sem1[0, 7] = 0.01 # r_brow
            sem1[0, 1] -= 0.02 - sem0[0, 6] - sem0[0, 7] # skin

        if mode == "newbrows":
            sem1 = sem0.clone()
            sem1[0, 6] += sem0[0, 6] # l_brow
            sem1[0, 7] += sem0[0, 7] # r_brow
            sem1[0, 1] += sem0[0, 6] + sem0[0, 7] # skin

        if mode == "hair":
            if sem0[0, 13] > 0.2: # too much hair
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            if sem0[0, 13] < 0.15: # not enough hair
                self.face_manipulation(savefile, bg_to_white=bg_to_white, mode=mode)
                return
            sem1 = sem0.clone()
            sem1[0, 13] = 0.4 # hair
            sem1[0, 1] -= 2/5 * (0.4 - sem0[0, 13]) # skin
            sem1[0, 0] -= 3/5 * (0.4 - sem0[0, 13]) # bg

        sem_cond = [sem1 * i / (self.vis_steps - 1) + sem0 * (1 - i / (self.vis_steps - 1)) for i in range(self.vis_steps)]
        # print("same", sem_cond[0] == sem0)
        sem_cond = torch.cat(sem_cond, dim=0)
        # print("orig", sem_cond[:, :4] * 10000)
        data["sem_cond"] = sem_cond
        data["sem_seg"] = data["sem_seg"].repeat(self.vis_steps, 1, 1, 1)
        data["z_seg"] = torch.randn(1, self.seg_opt.latent_dim).repeat(self.vis_steps, 1)
        fake_data = []
        batch_num = self.vis_steps // self.batch_size

        style = self.get_style()

        # save style
        if self.load_style:
            print("setting status to save_style")
            self.seg_img_model.img_model.netG.set_status("save_style")
            init_data = {k: v[[0]] if v.size(0) > 0 else v for k, v in data.items()}
            init_data["obj_dic"] = deepcopy(style)
            self.seg_img_model(data=init_data, mode='inference', log=True, as_list=False)
            obj_dic = init_data["obj_dic"]
            for i in force_mean_idx:
                obj_dic[str(i)]["ACE"] = style[str(i)]["ACE"]
        else:
            obj_dic = style
        # np.save("datasets/white.npy", obj_dic["0"]["ACE"].cpu().numpy())
        # obj_dic["0"]["ACE"] = torch.tensor(np.load("datasets/white.npy")).cuda() # set bg to white
        #print(ok)

        # use style
        print("setting status to use_style")
        self.seg_img_model.img_model.netG.set_status("use_style")
        for i in range(batch_num):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.vis_steps)
            batch_data = {k: v[start:end] if v.size(0) > 0 else v for k, v in data.items()}
            batch_data["obj_dic"] = obj_dic
            fake_data.append(self.seg_img_model(data=batch_data, mode='inference', log=True, as_list=False))
        fake_data = {k: torch.cat([fake[k] for fake in fake_data], dim=0) for k in fake_data[0].keys()}
        self.plot_face_interpolation(data, fake_data, savefile)

    def process_raw_sem(self, data):
        raw_sem = data["raw_sem_seg"].detach()
        raw = torch.argmax(raw_sem, dim=1, keepdim=True).cpu()
        raw[raw == self.opt.num_semantics] = -1
        return raw

    def plot_face_interpolation(self, data, fake_data, savefile):
        sem_seg = fake_data["sem_seg"]
        raw = self.process_raw_sem(fake_data)
        pred = torch.argmax(sem_seg, dim=1, keepdim=True).cpu()
        target = torch.argmax(data["sem_seg"], dim=1, keepdim=True).cpu()
        sem_img = (color_transfer(pred, self.colormap) + 1) / 2
        raw_img = (color_transfer(raw, self.colormap) + 1) / 2
        true_sem_img = (color_transfer(target, self.colormap) + 1) / 2
        sem_cond = data["sem_cond"].cpu()
        if self.addition_mode:
            sem_cond -= sem_cond[0].clone()
            sem_cond[sem_cond < 0] = 0
            fake_sem_cond = torch.mean(fake_data["raw_sem_seg"][:, :-1], dim=(2, 3)).cpu()
        else:
            fake_sem_cond = torch.mean(sem_seg, dim=(2, 3)).cpu()

        fake_img = (fake_data["img"].cpu() + 1) / 2
        img = (data["img"][0].cpu() + 1) / 2

        if self.opt.save_full_res:
            self.save_full_res(savefile, sem_cond, fake_sem_cond, [[true_sem_img[0]],
                                                                   [img],
                                                                   sem_img,
                                                                   fake_img,
                                                                   raw_img])

        def plot(i, get_fig=False):
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 15))

            ax1.imshow(true_sem_img[0].numpy().transpose(1, 2, 0).squeeze())
            ax1.set_axis_off()

            ax2.imshow(img.numpy().transpose(1, 2, 0).squeeze())
            ax2.set_axis_off()

            ax3.imshow(sem_img[i].numpy().transpose(1, 2, 0))
            ax3.set_axis_off()

            ax4.imshow(fake_img[i].numpy().transpose(1, 2, 0).squeeze())
            ax4.set_axis_off()

            if not self.seg_opt.fill_crop_only and not self.seg_opt.merged_activation:
                ax5.imshow(raw_img[i].numpy().transpose(1, 2, 0).squeeze())
            ax5.set_axis_off()

            x = np.array(range(self.opt.num_semantics))
            width = 0.5
            ax6.bar(x - width / 2, sem_cond[i], width, label='desired')
            ax6.bar(x + width / 2, fake_sem_cond[i], width, label='generated')
            ax6.xaxis.set_ticks(x)
            ax6.set_xticklabels(self.opt.semantic_labels, rotation='vertical')
            minor_ticks = np.array(range(-1, self.opt.num_semantics)) + 0.5
            ax6.set_xticks(minor_ticks, minor=True)
            ax6.xaxis.grid(which='major', alpha=1, c="white", lw=1)
            ax6.xaxis.grid(which='minor', alpha=1, c="grey", lw=1)
            ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax6.set_ylim(0, 0.1 + max(torch.max(sem_cond), torch.max(fake_sem_cond)))

            asp = np.diff(ax6.get_xlim())[0] / np.diff(ax6.get_ylim())[0]
            asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
            ax6.set_aspect(asp)

            fig.tight_layout()

            if get_fig:
                return fig

        def get_image(i):
            fig = plot(i, get_fig=True)
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

        imageio.mimsave(f'{self.opt.log_path}/{savefile}', [get_image(i) for i in tqdm(range(self.vis_steps), desc=savefile)], fps=8)


    def plot_ins_interpolation(self, fake_data, ins_cond, sem_cond, savefile):
        sem_seg = fake_data["sem_seg"]
        pred = torch.argmax(sem_seg, dim=1, keepdim=True).cpu()
        sem_img = (color_transfer(pred, self.colormap) + 1) / 2
        ins_cond = ins_cond.cpu()
        sem_cond = sem_cond.cpu()
        fake_sem_cond = torch.mean(sem_seg, dim=(2, 3)).cpu()
        fake_ins_cond = torch.sum(fake_data["ins_density"], dim=(2, 3)).cpu()
        img = (fake_data["img"].cpu() + 1) / 2

        if fake_data["ins_edge"].size(0) > 0:
            ins_img = fake_data["ins_edge"].cpu()
        elif fake_data["ins_center"].size(0) > 0:
            ins_offset = fake_data["ins_offset"].cpu()
            index = sem_seg.max(dim=1, keepdim=True)[1]
            seg_mc = torch.zeros_like(sem_seg).scatter_(1, index, 1.0)
            bg = (seg_mc[:, self.opt.things_idx].sum(dim=1) == 0)
            angle = (1 + torch.atan2(ins_offset[:, 1], ins_offset[:, 0]) / np.pi) / 2
            sat_norm = torch.min(10 * (torch.sqrt(ins_offset[:, 0] ** 2 + ins_offset[:, 1] ** 2)), torch.tensor([1.]))
            cmp = cm.get_cmap('hsv', 128)
            offset_rgba = cmp(angle.numpy())
            offset_rgb = torch.tensor(offset_rgba[:, :, :, :3]).float()
            offset_rgb = sat_norm.unsqueeze(-1) * offset_rgb + (1 - sat_norm).unsqueeze(-1) * torch.ones_like(offset_rgb)
            offset_rgb[bg] = torch.tensor([0., 0., 0.])
            offset_rgb = offset_rgb.permute(0, 3, 1, 2)
            center_mask = self.ins_refiner.get_peak_mask(fake_data["ins_center"].cpu()).float()
            ins_img = offset_rgb - 2 * center_mask
            ins_img[ins_img < 0] = 0.5

        if self.opt.save_full_res:
            self.save_full_res(savefile, sem_cond, fake_sem_cond, [img, sem_img, ins_img],
                               fake_ins=fake_ins_cond, real_ins=ins_cond)

        def plot(i, get_fig=False):
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 10))

            ax1.imshow(sem_img[i].numpy().transpose(1, 2, 0))
            ax1.set_axis_off()

            x = np.array(range(self.opt.num_semantics))
            width = 0.5
            ax2.bar(x - width / 2, sem_cond[i], width, label='desired')
            ax2.bar(x + width / 2, fake_sem_cond[i], width, label='generated')
            ax2.xaxis.set_ticks(x)
            ax2.set_xticklabels(self.opt.semantic_labels, rotation='vertical')
            minor_ticks = np.array(range(-1, self.opt.num_semantics)) + 0.5
            ax2.set_xticks(minor_ticks, minor=True)
            ax2.xaxis.grid(which='major', alpha=1, c="white", lw=1)
            ax2.xaxis.grid(which='minor', alpha=1, c="grey", lw=1)
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax2.set_ylim(0, 0.1 + max(torch.max(sem_cond), torch.max(fake_sem_cond)))

            asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
            asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
            ax2.set_aspect(asp)

            ax3.imshow(ins_img[i].numpy().transpose(1, 2, 0).squeeze())
            ax3.set_axis_off()

            x = np.array(range(self.opt.num_things))
            width = 0.5
            ax4.bar(x - width / 2, ins_cond[i], width, label='desired')
            ax4.bar(x + width / 2, fake_ins_cond[i], width, label='predicted')
            ax4.xaxis.set_ticks(x)
            ax4.set_xticklabels([self.opt.semantic_labels[k] for k in self.opt.things_idx], rotation='vertical')
            minor_ticks = np.array(range(-1, self.opt.num_semantics)) + 0.5
            ax4.set_xticks(minor_ticks, minor=True)
            ax4.xaxis.grid(which='major', alpha=1, c="white", lw=1)
            ax4.xaxis.grid(which='minor', alpha=1, c="grey", lw=1)
            ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax4.set_ylim(0, 0.1 + max(torch.max(ins_cond), torch.max(fake_ins_cond)))

            asp = np.diff(ax4.get_xlim())[0] / np.diff(ax4.get_ylim())[0]
            asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
            ax4.set_aspect(asp)

            ax5.imshow(img[i].numpy().transpose(1, 2, 0).squeeze())
            ax5.set_axis_off()

            ax6.set_axis_off()

            fig.tight_layout()

            if get_fig:
                return fig

        def get_image(i):
            fig = plot(i, get_fig=True)
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

        imageio.mimsave(f'{self.opt.log_path}/{savefile}', [get_image(i) for i in tqdm(range(self.batch_size), desc=savefile)], fps=8)

    def plot_sem_interpolation(self, fake_data, sem_cond, savefile):
        sem_seg = fake_data["sem_seg"]
        pred = torch.argmax(sem_seg, dim=1, keepdim=True).cpu()
        sem_img = (color_transfer(pred, self.colormap) + 1) / 2
        sem_cond = sem_cond.cpu()
        fake_sem_cond = torch.mean(sem_seg, dim=(2, 3)).cpu()
        img = (fake_data["img"].cpu() + 1) / 2
        sem_mask = fake_data["sem_mask"].cpu()
        spread = torch.sum(sem_mask, dim=1)
        spread_img = color_spread(spread, max_spread=5)
        raw_sem_seg = (sem_mask + 0.000001) / torch.sum(sem_mask + 0.000001, dim=1, keepdim=True)
        logprob = torch.log(raw_sem_seg + 0.00001)
        entropy_img = -torch.sum(torch.mul(raw_sem_seg, logprob), dim=1, keepdim=True)

        if self.opt.save_full_res:
            self.save_full_res(savefile, sem_cond, fake_sem_cond, [img,
                                                                   sem_img,
                                                                   spread_img,
                                                                   entropy_img])

        def plot(i, get_fig=False):
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(3, 2)
            ax1 = fig.add_subplot(gs[0, :])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            ax5 = fig.add_subplot(gs[2, 0])
            ax6 = fig.add_subplot(gs[2, 1])

            x = np.array(range(self.opt.num_semantics))
            width = 0.5
            ax1.bar(x - width / 2, sem_cond[i], width, label='desired')
            ax1.bar(x + width / 2, fake_sem_cond[i], width, label='generated')
            ax1.xaxis.set_ticks(x)
            ax1.set_xticklabels(self.opt.semantic_labels, rotation='vertical')
            minor_ticks = np.array(range(-1, self.opt.num_semantics)) + 0.5
            ax1.set_xticks(minor_ticks, minor=True)
            ax1.xaxis.grid(which='major', alpha=1, c="white", lw=1)
            ax1.xaxis.grid(which='minor', alpha=1, c="grey", lw=1)
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax1.set_ylim(0, 0.1 + max(torch.max(sem_cond), torch.max(fake_sem_cond)))

            # asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
            # asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
            # ax2.set_aspect(asp)

            ax3.imshow(sem_img[i].numpy().transpose(1, 2, 0))
            ax3.set_axis_off()

            ax4.imshow(img[i].numpy().transpose(1, 2, 0).squeeze())
            ax4.set_axis_off()

            ax5.imshow(spread_img[i].numpy().transpose(1, 2, 0).squeeze())
            ax5.set_axis_off()

            ax6.imshow(entropy_img[i].numpy().transpose(1, 2, 0).squeeze(), cmap='Greys_r')
            ax6.set_axis_off()

            fig.tight_layout()

            if get_fig:
                return fig

        def get_image(i):
            fig = plot(i, get_fig=True)
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

        imageio.mimsave(f'{self.opt.log_path}/{savefile}', [get_image(i) for i in tqdm(range(self.vis_steps), desc=savefile)], fps=8)


    def next_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.dataloader)
            return next(self.loader_iter)

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            self.dataset = create_dataset(self.opt, load_seg=True, load_img=True)
            self.dataloader, self.datasampler = engine.create_dataloader(self.dataset, self.opt.vis_dataloader_bs, self.opt.num_workers, True)
            self.loader_iter = iter(self.dataloader)
            is_main = self.opt.local_rank == 0
            logger = Logger(self.opt) if is_main else None
            self.seg_img_model_on_one_gpu = SegImgModel(self.seg_opt, self.img_opt, is_train=False, is_main=is_main, logger=logger)
            self.seg_img_model = engine.data_parallel(self.seg_img_model_on_one_gpu)
            self.seg_img_model.eval()
            self.colormap = create_colormap(self.opt)
            self.load_style = not self.opt.mean_style_only
            self.addition_mode = self.opt.addition_mode

            if self.opt.vis_random_style:
                self.style_codes = []
                for i in range(self.opt.num_semantics):
                    self.style_codes.append(glob(os.path.join(self.opt.extraction_path, 'style_codes/*', str(i), 'ACE.npy')))

            for i in range(self.opt.niter):
                if 'fixedsem' in self.opt.vis_method:
                    self.fixedsem_manipulation(f"fixedsem_manipulation_{i}.gif")
                if 'scrop' in self.opt.vis_method:
                    self.scrop_manipulation(f"scrop_manipulation_{i}.gif")
                if 'latent' in self.opt.vis_method:
                    self.latent_interpolation(f"latent_interpolation_{i}.gif")
                if 'cond' in self.opt.vis_method:
                    self.cond_interpolation(f"cond_interpolation_{i}.gif")
                if 'car' in self.opt.vis_method:
                    self.ins_manipulation(26, 0.25, 0.25, 0.5, 20, f"car_manipulation_{i}.gif")
                if 'person' in self.opt.vis_method:
                    self.ins_manipulation(24, 0.25, 0.25, 1, 20, f"person_manipulation_{i}.gif")
                if 'face' in self.opt.vis_method:
                    self.face_manipulation(f"face_manipulation_{i}.gif")
                if 'earrings' in self.opt.vis_method:
                    self.face_manipulation(f"earrings_manipulation_{i}.gif", mode="earrings")
                if 'hair' in self.opt.vis_method:
                    self.face_manipulation(f"hair_manipulation_{i}.gif", mode="hair")
                if 'rebald' in self.opt.vis_method:
                    self.face_manipulation(f"bald_manipulation_{i}.gif", mode="bald")
                if 'unbald' in self.opt.vis_method:
                    self.face_manipulation(f"unbald_manipulation_{i}.gif", mode="unbald")
                if 'hat' in self.opt.vis_method:
                    self.face_manipulation(f"hat_manipulation_{i}.gif", mode="hat")
                if 'eyebrows' in self.opt.vis_method:
                    self.face_manipulation(f"eyebrows_manipulation_{i}.gif", mode="eyebrows")
                if 'teeth' in self.opt.vis_method:
                    self.face_manipulation(f"teeth_manipulation_{i}.gif", mode="teeth")
                if 'glasses' in self.opt.vis_method:
                    self.face_manipulation(f"glasses_manipulation_{i}.gif", mode="glasses")
                if 'openeyes' in self.opt.vis_method:
                    self.face_manipulation(f"eyes_manipulation_{i}.gif", mode="openeyes")
                if 'newbrows' in self.opt.vis_method:
                    self.face_manipulation(f"newbrows_manipulation_{i}.gif", mode="newbrows")
                if 'morebrows' in self.opt.vis_method:
                    self.face_manipulation(f"morebrows_manipulation_{i}.gif", mode="morebrows")
                if 'nose' in self.opt.vis_method:
                    self.face_manipulation(f"nose_manipulation_{i}.gif", mode="nose")
                if 'skin' in self.opt.vis_method:
                    self.face_manipulation(f"skin_manipulation_{i}.gif", mode="skin")

            print('Visualization was successfully finished.')

if __name__ == "__main__":
    opt = Options().parse(load_seg_generator=True, load_img_generator=True, save=True)
    Visualizer(opt).run()





