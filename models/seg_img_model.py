import torch

from models.img_generator.models.img_model import ImgModel as ImgModelGen
from models.img_style_generator.models.img_model import ImgModel as ImgModelSty
from models.seg_generator.models.seg_model import SegModel as SegModelGen
from models.seg_completor.models.seg_model import SegModel as SegModelCom


class SegImgModel(torch.nn.Module):
    def __init__(self, opt_seg, opt_img, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt_seg = opt_seg
        self.opt_img = opt_img
        self.is_main = is_main

        if opt_seg.seg_type == "generator":
            self.seg_model = SegModelGen(self.opt_seg, is_train=is_train, is_main=is_main, logger=logger)
        elif opt_seg.seg_type == "completor":
            self.seg_model = SegModelCom(self.opt_seg, is_train=is_train, is_main=is_main, logger=logger)
        if opt_img.img_type == "generator":
            self.img_model = ImgModelGen(self.opt_img, is_train=is_train, is_main=is_main, logger=logger)
        elif opt_img.img_type == "style_generator":
            self.img_model = ImgModelSty(self.opt_img, is_train=is_train, is_main=is_main, logger=logger)

        self.logger = logger if self.is_main else None

        self.update_seg_model = not opt_seg.no_update_seg_model
        self.update_img_model_fake = True
        self.update_img_model_real = True
        self.fake_from_fake_dis = opt_img.fake_from_fake_dis
        self.fake_from_real_dis = opt_img.fake_from_real_dis


    def forward(self, data, tgt_data=None, fake=None, mode='', log=False, global_iteration=None, as_list=True):
        if mode == 'generator':
            g_loss, fake = self.compute_generator_loss(data, tgt_data, log, global_iteration)
            return g_loss, fake

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(data, tgt_data, fake, log, global_iteration)
            return d_loss

        elif mode == 'd2':
            d_loss = self.compute_d2_loss(data, tgt_data, fake, log, global_iteration)
            return d_loss

        elif mode == 'inference':
            with torch.no_grad():
                fake = self.generate_fake(data, log, global_iteration)
            if as_list:
                return fake
            else:
                fake_seg, fake_img, _ = fake
                fake_data = {}
                fake_data.update(fake_seg)
                fake_data.update(fake_img)
                return fake_data

        else:
            raise ValueError(f"mode '{mode}' is invalid")


    def save_model(self, global_iteration, latest, d2_only=False):
        if d2_only:
            self.img_model.save_d2(global_iteration, latest=latest)
        else:
            self.img_model.save_model(global_iteration, latest=latest)
            self.seg_model.save_model(global_iteration, latest=latest)


    def compute_generator_loss(self, data, tgt_data, log, global_iteration):
        loss = torch.tensor([0.]).cuda()
        fake_img_fake = {}
        fake_img_real = {}

        # fake seg
        if self.update_seg_model:
            loss_seg_gen, fake_seg = self.seg_model(data, mode='generator', log=log, hard=True,
                                                    global_iteration=global_iteration)
            loss += loss_seg_gen
        else:
            with torch.no_grad():
                fake_seg = self.seg_model(data, mode='inference', log=log, hard=True,
                                          global_iteration=global_iteration)

        # fake img from fake seg
        if self.update_img_model_fake:
            fake_seg["img"] = data["img"]
            loss_img_gen_fake, fake_img_fake = self.img_model(fake_seg, tgt_data=tgt_data, mode='generator', log=log,
                                                              global_iteration=global_iteration, suffix="_from_fake",
                                                              dis=self.fake_from_fake_dis, has_gt=False)
            loss += loss_img_gen_fake

        # fake img from real seg
        if self.update_img_model_real:
            loss_img_gen_real, fake_img_real = self.img_model(data, tgt_data=tgt_data, mode='generator', log=log,
                                                              global_iteration=global_iteration, suffix="_from_real",
                                                              dis=self.fake_from_real_dis, has_gt=True)
            loss += loss_img_gen_real

        return loss, [fake_seg, fake_img_fake, fake_img_real]


    def compute_discriminator_loss(self, data, tgt_data, fake, log, global_iteration):
        loss = torch.tensor([0.]).cuda()
        fake_seg, fake_img_fake, fake_img_real = fake

        if self.update_seg_model:
            loss_seg_dis = self.seg_model(data, fake_data=fake_seg, mode='discriminator', log=log,
                                          global_iteration=global_iteration)
            loss += loss_seg_dis

        if self.update_img_model_fake:
            fake_data = fake_seg.copy()
            fake_data.update(fake_img_fake)
            loss_img_dis_fake = self.img_model(data, tgt_data=tgt_data, fake_data=fake_data, mode='discriminator',
                                               global_iteration=global_iteration, suffix="_from_fake", has_gt=False,
                                               dis=self.fake_from_fake_dis, log=log)
            loss += loss_img_dis_fake

        if self.update_img_model_real:
            fake_data = data.copy()
            fake_data.update(fake_img_real)
            loss_img_dis_real = self.img_model(data, tgt_data=tgt_data, fake_data=fake_data, mode='discriminator',
                                               global_iteration=global_iteration, suffix="_from_real", has_gt=True,
                                               dis=self.fake_from_real_dis, log=log)
            loss += loss_img_dis_real

        return loss


    def compute_d2_loss(self, data, tgt_data, fake, log, global_iteration):
        loss = torch.tensor([0.]).cuda()
        fake_seg, fake_img_fake, fake_img_real = fake

        fake_data = fake_seg.copy()
        fake_data.update(fake_img_fake)
        loss_img_dis_fake = self.img_model(data, tgt_data=tgt_data, fake_data=fake_data, mode='discriminator', log=log,
                                           global_iteration=global_iteration, suffix="_from_fake",
                                           dis=self.fake_from_fake_dis)
        loss += loss_img_dis_fake

        return loss


    def generate_fake(self, data, log, global_iteration):
        fake_seg = self.seg_model(data, mode='inference', hard=True)
        data_for_img = fake_seg.copy()
        data_for_img['img'] = data['img'] if 'img' in data else torch.tensor([])
        data_for_img['obj_dic'] = data['obj_dic'] if 'obj_dic' in data else None
        fake_img = self.img_model(data_for_img, mode='inference')
        return [fake_seg, fake_img, None]