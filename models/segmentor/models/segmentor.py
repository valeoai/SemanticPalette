import torch

from .deeplabv3 import Seg_Model as Deeplabv3
from .pspnet import Seg_Model as PSPNet
from models import load_network, save_network, print_network
from models.segmentor.loss.criterion import CriterionDSN, CriterionOhemDSN
from models.segmentor.models.advent.utils.func import prob_2_entropy, bce_loss
from models.segmentor.models.advent.model.discriminator import get_fc_discriminator

class Segmentor(torch.nn.Module):
    def __init__(self, opt, is_train=True, is_main=True, logger=None, distributed=False):
        super().__init__()
        self.opt = opt
        self.is_main = is_main

        self.logger = logger if is_main else None

        self.netS, self.netD_final, self.netD_inter = self.initialize_network(is_train, distributed=distributed)
        height = self.opt.fixed_crop[0] if self.opt.fixed_crop is not None else int(self.opt.dim)
        width = self.opt.fixed_crop[1] if self.opt.fixed_crop is not None else int(self.opt.dim * self.opt.aspect_ratio)
        self.interpolate = torch.nn.Upsample(size=(height, width), mode='bilinear', align_corners=True)
        self.softmax = torch.nn.Softmax2d()

        self.ignore_index = 255

        if is_train:
            self.opt_s, self.opt_d = self.create_optimizer(self.opt)
            if self.opt.ohem:
                segmentation_loss = CriterionOhemDSN(ignore_index=self.ignore_index, thresh=self.opt.ohem_thres,
                                                     min_kept=self.opt.ohem_keep, reduction='none')
            else:
                segmentation_loss = CriterionDSN(ignore_index=self.ignore_index, reduction='none')
            self.segmentation_loss = segmentation_loss.cuda()


    def forward(self, data, target_data={}, pred_data={}, global_iteration=0, log=False, mode='', hard=True, suffix=''):
        img, src_sem_seg = self.preprocess_input(data)
        tgt_img, _ = self.preprocess_input(target_data)
        pred_seg, pred_seg_tgt = self.preprocess_input(pred_data, is_pred=True)

        if mode == 'segmentor':
            loss, pred_seg, delta = self.compute_segmentation_loss(src_sem_seg, img, hard, log, global_iteration, suffix)
            return loss, pred_seg, delta
        
        if mode == 'segmentor_advent':
            loss, pred_seg, delta = self.compute_advent_segmentation_loss(src_sem_seg, img, tgt_img, hard, log, global_iteration)
            return loss, pred_seg, delta

        if mode == 'discriminator_advent':
            loss = self.compute_advent_discriminator_loss(pred_seg, pred_seg_tgt, log, global_iteration)
            return loss

        if mode == 'inference':
            pred_seg = self.predict(img, hard, log, global_iteration)
            return pred_seg


    def preprocess_input(self, data, is_pred=False):
        if is_pred:
            data["sem_seg"] = data["sem_seg"].detach() if "sem_seg" in data else  torch.Tensor([])
            data["sem_seg_inter"] = data["sem_seg_inter"].detach() if "sem_seg_inter" in data else torch.Tensor([])
            data["sem_seg_tgt"] = data["sem_seg_tgt"].detach() if "sem_seg_tgt" in data else torch.Tensor([])
            data["sem_seg_inter_tgt"] = data["sem_seg_inter_tgt"].detach() if "sem_seg_inter_tgt" in data else torch.Tensor([])
            pred_seg = {"sem_seg": data["sem_seg"], "sem_seg_inter": data["sem_seg_inter"]}
            pred_seg_tgt = {"sem_seg": data["sem_seg_tgt"], "sem_seg_inter": data["sem_seg_inter_tgt"]}
            return pred_seg, pred_seg_tgt
        else:
            data["img"] = data["img"].cuda() if "img" in data else torch.Tensor([])
            data["sem_seg"] = data["sem_seg"].cuda() if "sem_seg" in data else torch.Tensor([])
            img = data["img"]
            src_sem_seg = data["sem_seg"]
            return img, src_sem_seg


    def flatten_sem(self, sem_seg):
        if self.opt.segment_eval_classes_only:
            reduced_sem = self.reduce_sem(sem_seg)
            no_eval_mask = torch.max(reduced_sem, dim=1)[0] == 0
            max_idx = reduced_sem.max(dim=1)[1]
            max_idx[no_eval_mask] = self.ignore_index
        else:
            max_idx = sem_seg.max(dim=1)[1]
        return max_idx


    def expand_sem(self, sem_seg):
        if self.opt.segment_eval_classes_only:
            s = sem_seg.shape
            expanded_sem = torch.zeros(s[0], self.opt.num_semantics, s[2], s[3]).cuda()
            expanded_sem[:, self.opt.eval_idx] = sem_seg
            return expanded_sem
        else:
            return sem_seg


    def reduce_sem(self, sem_seg):
        if self.opt.segment_eval_classes_only:
            reduced_sem = sem_seg[:, self.opt.eval_idx]
            return reduced_sem
        else:
            return sem_seg


    def initialize_network(self, is_train, distributed=False):
        num_classes = len(self.opt.eval_idx) if self.opt.segment_eval_classes_only else self.opt.num_semantics
        if self.opt.model == 'pspnet':
            netS = PSPNet(num_classes=num_classes, distributed=distributed).cuda()
        elif self.opt.model == 'deeplabv3':
            netS = Deeplabv3(num_classes=num_classes, distributed=distributed).cuda()

        if is_train and self.opt.advent:
            netD_final = get_fc_discriminator(num_classes=num_classes).cuda()
            netD_inter = get_fc_discriminator(num_classes=num_classes).cuda() if self.opt.advent_multi else None
        else:
            netD_final = None
            netD_inter = None

        if self.is_main:
            if self.opt.pretrained_path is not None:
                netS = self.load_model(netS)
            else:
                netS = load_network(netS, "segmentor", self.opt)
            print_network(netS)
            if is_train and self.opt.advent:
                if self.opt.cont_train:
                    netD_final = load_network(netD_final, "advent_d_final", self.opt)
                print_network(netD_final)
                if self.opt.advent_multi:
                    if self.opt.cont_train:
                        netD_inter = load_network(netD_inter, "advent_d_inter", self.opt)
                    print_network(netD_inter)

        return netS, netD_final, netD_inter


    def load_model(self, net):
        saved_state_dict = torch.load(self.opt.pretrained_path)
        new_params = net.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc' or not self.opt.not_restore_last:
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        net.load_state_dict(new_params)
        return net


    def save_model(self, global_iteration, latest):
        save_network(self.netS, "segmentor", global_iteration, self.opt, latest=latest)
        if self.opt.advent:
            save_network(self.netD_final, "advent_d_final", global_iteration, self.opt, latest=latest)
            if self.opt.advent_multi:
                save_network(self.netD_inter, "advent_d_inter", global_iteration, self.opt, latest=latest)


    def create_optimizer(self, opt):
        if opt.optimizer == "sgd":
            params = filter(lambda p: p.requires_grad, self.netS.parameters())
            opt_s = torch.optim.SGD([{'params': params, 'lr': self.opt.lr}], lr=self.opt.lr, momentum=self.opt.momentum,
                                    weight_decay=self.opt.weight_decay)
        else:
            raise NotImplementedError
        
        if self.opt.advent:
            params = list(self.netD_final.parameters())
            if self.opt.advent_multi:
                params += list(self.netD_inter.parameters())
            opt_d = torch.optim.Adam(params, lr=self.opt.advent_lr, betas=(0.9, 0.99))
        else:
            opt_d = None
            
        return opt_s, opt_d


    def compute_delta(self, unreduced_loss, src_sem_seg):
        with torch.no_grad():
            index = src_sem_seg.max(dim=1, keepdim=True)[1]
            src_sem_seg = torch.zeros_like(src_sem_seg).scatter_(1, index, 1.0)
            batch_delta = unreduced_loss.mean(dim=(1, 2))
            class_loss = (unreduced_loss.unsqueeze(1) * src_sem_seg).sum(dim=(2, 3))
            class_tot = src_sem_seg.sum(dim=(2, 3))
            class_surface = class_tot.sum(dim=0)
            class_num = (class_tot > 0).float().sum(dim=0)
            class_delta = class_loss / class_tot.clamp(min=1)
            class_delta = class_delta.sum(dim=0)
        return {"batch_delta": batch_delta, "class_delta": class_delta, "class_surface": class_surface, "class_num": class_num}


    def compute_segmentation_loss(self, src_sem_seg, img, hard, log, global_iteration, suffix):
        pred_seg = self.predict(img=img, hard=False, log=False, global_iteration=global_iteration)
        flat_src = self.flatten_sem(src_sem_seg)
        unreduced_loss = self.segmentation_loss([self.reduce_sem(pred_seg["sem_seg"]), self.reduce_sem(pred_seg["inter_sem_seg"])], flat_src)
        delta = self.compute_delta(unreduced_loss, src_sem_seg)
        loss = unreduced_loss.mean()

        if self.logger:
            # log scalars every step
            self.logger.log_scalar(f"segmentor/segmentation{suffix}", loss.cpu(), global_iteration)
            # log images every few steps
            if log:
                self.logger.log_semantic_seg(f"segmentor/real{suffix}", src_sem_seg[:16].float().cpu(), 4, global_iteration)
                self.logger.log_semantic_seg(f"segmentor/pred{suffix}", pred_seg["sem_seg"][:16].float().cpu(), 4, global_iteration)
                self.logger.log_img(f"segmentor/img{suffix}", img[:16].float().cpu(), 4, global_iteration, normalize=True, range=(-1, 1))

        if hard:
            index = pred_seg["sem_seg"].max(dim=1, keepdim=True)[1]
            x_pred_sem_seg = torch.zeros_like(pred_seg["sem_seg"]).scatter_(1, index, 1.0)
            return loss, {"sem_seg": x_pred_sem_seg, "inter_sem_seg": pred_seg["inter_sem_seg"]}, delta
        else:
            return loss, pred_seg, delta
        
    
    def compute_advent_segmentation_loss(self, src_sem_seg, img, tgt_img, hard, log, global_iteration):
        src_label = 0

        # segmentation loss
        # train with src
        pred_seg = self.predict(img=img, hard=False, log=False, global_iteration=global_iteration)
        flat_src = self.flatten_sem(src_sem_seg)
        unreduced_loss = self.segmentation_loss([self.reduce_sem(pred_seg["sem_seg"]), self.reduce_sem(pred_seg["inter_sem_seg"])], flat_src)
        delta = self.compute_delta(unreduced_loss, src_sem_seg)
        loss_seg = unreduced_loss.mean()
        loss_s = loss_seg

        # adv loss
        for param in self.netD_final.parameters():
            param.requires_grad = False
        if self.opt.advent_multi:
            for param in self.netD_inter.parameters():
                param.requires_grad = False
        # train with tgt
        pred_seg_tgt = self.predict(img=tgt_img, hard=False, log=False, global_iteration=global_iteration)
        if self.opt.advent_multi:
            ent_tgt_inter = prob_2_entropy(self.softmax(pred_seg_tgt["inter_sem_seg"]))
            d_out_inter = self.netD_inter(ent_tgt_inter)
            loss_adv_tgt_inter = bce_loss(d_out_inter, src_label)
        else:
            loss_adv_tgt_inter = 0
        ent_tgt_final = prob_2_entropy(self.softmax(pred_seg_tgt["sem_seg"]))
        d_out_final = self.netD_final(ent_tgt_final)
        loss_adv_tgt_final = bce_loss(d_out_final, src_label)
        loss_adv = self.opt.advent_lambda_adv_final * loss_adv_tgt_final + \
                   self.opt.advent_lambda_adv_inter * loss_adv_tgt_inter
        loss_s += loss_adv

        ent_tgt = torch.sum(ent_tgt_final, dim=1, keepdim=True)

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("segmentor/segmentation", loss_seg, global_iteration)
            self.logger.log_scalar("segmentor/advent_adv_inter", loss_adv_tgt_inter, global_iteration)
            self.logger.log_scalar("segmentor/advent_adv_final", loss_adv_tgt_final, global_iteration)
            self.logger.log_scalar("segmentor/advent_adv", loss_adv, global_iteration)
            self.logger.log_scalar("segmentor/advent_ent_tgt", torch.mean(ent_tgt), global_iteration)
            # log images every few steps
            if log:
                self.logger.log_img("segmentor/entropy_tgt", ent_tgt[:16].float().cpu(), 4, global_iteration)
                self.logger.log_semantic_seg("segmentor/src", src_sem_seg[:16].float().cpu(), 4, global_iteration)
                self.logger.log_semantic_seg("segmentor/pred_tgt", pred_seg_tgt["sem_seg"][:16].float().cpu(), 4, global_iteration)
                self.logger.log_semantic_seg("segmentor/pred_src", pred_seg["sem_seg"][:16].float().cpu(), 4, global_iteration)
                self.logger.log_img("segmentor/img_src", img[:16].float().cpu(), 4, global_iteration, normalize=True, range=(-1, 1))
                self.logger.log_img("segmentor/img_tgt", tgt_img[:16].float().cpu(), 4, global_iteration, normalize=True, range=(-1, 1))

        if hard:
            index = pred_seg["sem_seg"].max(dim=1, keepdim=True)[1]
            x_pred_sem_seg = torch.zeros_like(pred_seg["sem_seg"]).scatter_(1, index, 1.0)
            pred_data = {"sem_seg": x_pred_sem_seg, "inter_sem_seg": pred_seg["inter_sem_seg"]}
        else:
            pred_data = pred_seg
        pred_data["sem_seg_tgt"] = pred_seg_tgt["sem_seg"]
        pred_data["inter_sem_seg_tgt"] = pred_seg_tgt["inter_sem_seg"]
        return loss_s, pred_data, delta

    def compute_advent_discriminator_loss(self, pred_seg, pred_seg_tgt, log, global_iteration):
        src_label = 0
        tgt_label = 1

        # dis loss
        loss_d = 0
        for param in self.netD_final.parameters():
            param.requires_grad = True
        if self.opt.advent_multi:
            for param in self.netD_inter.parameters():
                param.requires_grad = True
        # train with src
        if self.opt.advent_multi:
            pred_seg["inter_sem_seg"] = pred_seg["inter_sem_seg"].detach()
            ent_src_inter = prob_2_entropy(self.softmax(pred_seg["inter_sem_seg"]))
            d_out_inter = self.netD_inter(ent_src_inter)
            loss_dis_src_inter = bce_loss(d_out_inter, src_label) / 2
            loss_d += loss_dis_src_inter
        else:
            loss_dis_src_inter = 0
        pred_seg["sem_seg"] = pred_seg["sem_seg"].detach()
        ent_src_final = prob_2_entropy(self.softmax(pred_seg["sem_seg"]))
        d_out_final = self.netD_final(ent_src_final)
        loss_dis_src_final = bce_loss(d_out_final, src_label) / 2
        loss_d += loss_dis_src_final
        # train with target
        if self.opt.advent_multi:
            pred_seg_tgt["inter_sem_seg"] = pred_seg_tgt["inter_sem_seg"].detach()
            ent_tgt_inter = prob_2_entropy(self.softmax(pred_seg_tgt["inter_sem_seg"]))
            d_out_inter = self.netD_inter(ent_tgt_inter)
            loss_dis_tgt_inter = bce_loss(d_out_inter, tgt_label) / 2
            loss_d += loss_dis_tgt_inter
        else:
            loss_dis_tgt_inter = 0
        pred_seg_tgt["sem_seg"] = pred_seg_tgt["sem_seg"].detach()
        ent_tgt_final = prob_2_entropy(self.softmax(pred_seg_tgt["sem_seg"]))
        d_out_final = self.netD_final(ent_tgt_final)
        loss_dis_tgt_final = bce_loss(d_out_final, tgt_label) / 2
        loss_d += loss_dis_tgt_final

        ent_src = torch.sum(ent_src_final, dim=1, keepdim=True)
        loss_dis_inter = loss_dis_tgt_inter + loss_dis_src_inter
        loss_dis_final = loss_dis_tgt_final + loss_dis_src_final

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("segmentor/advent_dis_inter", loss_dis_inter, global_iteration)
            self.logger.log_scalar("segmentor/advent_dis_final", loss_dis_final, global_iteration)
            self.logger.log_scalar("segmentor/advent_dis", loss_d, global_iteration)
            self.logger.log_scalar("segmentor/advent_ent_src", torch.mean(ent_src), global_iteration)
            if log:
                self.logger.log_img("segmentor/entropy_src", ent_src[:16].cpu(), 4, global_iteration)

        return loss_d

    def predict(self, img, hard, log, global_iteration):
        preds = self.netS(img)
        preds[0] = self.expand_sem(self.interpolate(preds[0]))
        preds[1] = self.expand_sem(self.interpolate(preds[1]))

        if self.logger and log:
            self.logger.log_semantic_seg("segmentor/pred", preds[0][:16], 4, global_iteration)

        if hard:
            index = preds[0].max(dim=1, keepdim=True)[1]
            x_pred_sem_seg = torch.zeros_like(preds[0]).scatter_(1, index, 1.0)
            return {"sem_seg": x_pred_sem_seg, "inter_sem_seg":preds[1]}
        else:
            return {"sem_seg": preds[0], "inter_sem_seg": preds[1]}


