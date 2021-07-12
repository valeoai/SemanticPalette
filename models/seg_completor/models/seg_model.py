import random
import torch
import torch.nn.functional as F

from ..models.progressive import ProGANGenerator
from ..models.progressive import ProGANDiscriminator
from ..models.style import StyleGANGenerator
from ..modules.gan_loss import ImprovedWGANLoss
from ..modules.instance_refiner import InstanceRefiner
from tools.utils import to_cuda
from models import load_network, save_network, print_network


class SegModel(torch.nn.Module):
    def __init__(self, opt, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt = opt
        self.is_main = is_main
        self.is_train = is_train

        self.netG, self.netD = self.initialize_networks(is_train)

        if is_train:
            self.opt_g, self.opt_d = self.create_optimizers(self.opt)
            self.gan_loss = ImprovedWGANLoss(self.netD)

        self.logger = logger if self.is_main else None

        self.ins_refiner = InstanceRefiner(self.opt)


    def forward(self, data, fake_data={}, interpolate=False, alpha=None, mode='', log=False, hard=True,
                global_iteration=None):
        z, real_seg, real_cond = self.preprocess_input(data)
        _, fake_seg, _ = self.preprocess_input(fake_data, is_fake=True)

        if mode == 'generator':
            g_loss, fake_seg = self.compute_generator_loss(real_cond, real_seg, z, interpolate, alpha, hard, log, global_iteration)
            fake_seg = self.postprocess_output(fake_seg)
            return g_loss, fake_seg

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(real_seg, fake_seg, interpolate, alpha, log, global_iteration)
            return d_loss

        elif mode == 'inference':
            fake_seg = self.generate_fake(real_cond, real_seg, z, interpolate, alpha, hard, log, global_iteration)
            fake_seg = self.postprocess_output(fake_seg)
            return fake_seg

        else:
            raise ValueError(f"mode '{mode}' is invalid")


    def postprocess_output(self, seg):
        if self.opt.dim != self.opt.seg_dim:
            size = (int(self.opt.dim), int(self.opt.aspect_ratio * self.opt.dim))
            mode = 'bilinear' if self.opt.discretization == "none" or self.opt.bilimax else 'nearest'
            seg = {k: self.resize(v, size, mode=mode) for k, v in seg.items()}
            if self.opt.bilimax:
                index = seg["sem_seg"].max(1, keepdim=True)[1]
                seg["sem_seg"] = torch.zeros_like(seg["sem_seg"]).scatter_(1, index, 1.0)
        return seg


    def resize(self, t, size, mode='nearest'):
        if size is not None and not 0 in t.size():
            return torch.nn.functional.interpolate(t, size=size, mode=mode)
        else:
            return t


    def preprocess_input(self, data, is_fake=False):
        size = (int(self.opt.seg_dim), int(self.opt.aspect_ratio * self.opt.seg_dim)) if self.opt.dim != self.opt.seg_dim else None
        data["z_seg"] = to_cuda(data, "z_seg")
        data["sem_seg"] = to_cuda(data, "sem_seg")
        data["ins_center"] = to_cuda(data, "ins_center")
        data["ins_offset"] = to_cuda(data, "ins_offset")
        data["ins_edge"] = to_cuda(data, "ins_edge")
        data["ins_density"] = to_cuda(data, "ins_density")
        data["sem_cond"] = to_cuda(data, "sem_cond")
        data["ins_cond"] = to_cuda(data, "ins_cond")

        if is_fake:
            data["sem_seg"] = data["sem_seg"].detach()
            data["ins_center"] = data["ins_center"].detach()
            data["ins_offset"] = data["ins_offset"].detach()
            data["ins_edge"] = data["ins_edge"].detach()
            data["ins_density"] = data["ins_density"].detach()

        z = data["z_seg"]
        seg = {'sem_seg': self.resize(data["sem_seg"], size),
               'ins_center': self.resize(data["ins_center"], size),
               'ins_offset': self.resize(data["ins_offset"], size),
               'ins_edge': self.resize(data["ins_edge"], size),
               'ins_density': self.resize(data["ins_density"], size)}
        cond = {'sem_cond': data["sem_cond"],
                'ins_cond': data["ins_cond"]}

        return z, seg, cond


    def initialize_networks(self, is_train):
        if self.opt.model == 'progressive':
            netG = ProGANGenerator(self.opt).cuda()
            netD = ProGANDiscriminator(self.opt).cuda() if is_train else None
        elif self.opt.model == 'style':
            netG = StyleGANGenerator(self.opt).cuda()
            netD = ProGANDiscriminator(self.opt).cuda() if is_train else None

        if self.is_main:
            netG = load_network(netG, "seg_g", self.opt)
            print_network(netG)
            if is_train:
                netD = load_network(netD, "seg_d", self.opt)
                print_network(netD)

        netG.res = self.opt.seg_dim
        if netD:
            netD.res = self.opt.seg_dim

        return netG, netD


    def save_model(self, global_iteration, latest):
        save_network(self.netG, "seg_g", global_iteration, self.opt, latest=latest)
        save_network(self.netD, "seg_d", global_iteration, self.opt, latest=latest)


    def create_optimizers(self, opt):
        if opt.optimizer == "adam":
            opt_g = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            opt_d = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        else:
            raise NotImplementedError
        return opt_g, opt_d


    def transform_sem_cond(self, sem, cond):
        crop_prop = None
        if self.opt.vertical_sem_crop:
            index = sem.max(1, keepdim=True)[1]
            sem = torch.zeros_like(sem).scatter_(1, index, 1.0)
            for i in range(sem.shape[0]):
                min_crop = self.opt.min_sem_crop
                l = round((0.5 + random.random() * 0.5) * sem.shape[-1])
                start = round((sem.shape[-1] - l) * random.random())
                stop = start + l
                if self.opt.cond_seg is not None:
                    target_cond_selection = cond["sem_cond"][i][self.opt.bias_sem] * self.opt.bias_mul
                    target_cond_tot = torch.sum(target_cond_selection)
                    cond["sem_cond"][i] = torch.mean(sem[i, :, :, start:stop], dim=(1, 2))
                    if target_cond_tot:
                        cond["sem_cond"][i][self.opt.bias_sem] += target_cond_selection
                        cond["sem_cond"][i] /= (1 + target_cond_tot)
                    cond["sem_cond"][i] *= l / sem.shape[-1]
                sem[i, :, :, start:stop] = 1. / self.opt.num_semantics
        if len(self.opt.sem_label_crop) > 0:
            index = sem.max(1, keepdim=True)[1]
            sem = torch.zeros_like(sem).scatter_(1, index, 1.0)
            mask = torch.zeros_like(index, dtype=bool)
            for i in self.opt.sem_label_crop:
                mask += sem[:, [i]] == 1.0
            crop_prop = torch.mean(mask.float(), dim=(2, 3))
            if self.opt.edge_cond:
                edge = torch.zeros_like(mask)
                edge[:, 0, :, 1:] = edge[:, 0, :, 1:] | (mask[:, 0, :, 1:] != mask[:, 0, :, :-1])
                edge[:, 0, :, :-1] = edge[:, 0, :, :-1] | (mask[:, 0, :, 1:] != mask[:, 0, :, :-1])
                edge[:, 0, 1:, :] = edge[:, 0, 1:, :] | (mask[:, 0, 1:, :] != mask[:, 0, :-1, :])
                edge[:, 0, :-1, :] = edge[:, 0, :-1, :] | (mask[:, 0, 1:, :] != mask[:, 0, :-1, :])
                crop_edge = edge & ~mask
                cond["sem_cond"] = torch.mean(sem * crop_edge, dim=(2, 3))
                cond_sum = torch.sum(cond["sem_cond"], dim=1, keepdim=True)
                cond_sum[cond_sum == 0] = 1
                cond["sem_cond"] /= cond_sum
                cond["sem_cond"] *= torch.mean(mask.float() + crop_edge.float(), dim=(2, 3))
            sem[mask.expand_as(sem)] = 1. / self.opt.num_semantics

        if self.opt.cond_seg is not None:
            if self.opt.switch_cond:
                target_cond = cond["sem_cond"]
                real_cond = torch.mean(sem, dim=(2, 3))
                if self.opt.random_soft_mix and random.random() > 0.5:
                    target_cond = 0.75 * real_cond + 0.25 * target_cond
                elif self.opt.random_linear:
                    alpha = torch.rand((target_cond.shape[0], 1)).to(target_cond.get_device())
                    target_cond = alpha * target_cond + (1. - alpha) * real_cond
                delta = target_cond - real_cond
                inter = torch.sum(torch.min(target_cond, real_cond), dim=1)
                cond["inter_cond"] = inter
                cond["delta_cond"] = delta.clone()
                delta[delta < 0.001] = 0
                cond["sem_cond"] = target_cond if self.opt.merged_activation else delta
            if len(self.opt.sem_label_ban) > 0:
                sem_label_not_ban = [i for i in range(self.opt.num_semantics) if i not in self.opt.sem_label_ban]
                cond["sem_cond"] = cond["sem_cond"][:, sem_label_not_ban]
                if self.opt.weight_cond_crop:
                    assert crop_prop is not None
                    cond["sem_cond"] /= torch.sum(cond["sem_cond"], dim=1, keepdim=True)
                    cond["sem_cond"] *= crop_prop
            if not self.opt.fill_crop_only and not self.opt.merged_activation:
                bg_cond = 1. - torch.sum(cond["sem_cond"], dim=1, keepdim=True)
                cond["sem_cond"] = torch.cat([cond["sem_cond"], bg_cond], dim=1)
            cond["sem_cond"][cond["sem_cond"] < 0] = 0
            cond["sem_cond"] /= torch.sum(cond["sem_cond"], dim=1, keepdim=True)
        # print("prepro", cond["sem_cond"][:, :4] * 10000)
        return sem, cond


    def compute_generator_loss(self, real_cond, real_seg, z, interpolate, alpha, hard, log, global_iteration):
        loss_gen_things = None
        sem, real_cond = self.transform_sem_cond(real_seg["sem_seg"].clone(), real_cond)
        if interpolate:
            fake_segs = self.netG.interpolate(z, alpha, sem, cond=real_cond)
        else:
            fake_segs = self.netG(z, sem, cond=real_cond)

        if not "inter" in self.opt.cond_mode:
            fake_segs = [fake_segs[-1]]

        x_fake_segs = [self.to_discrete(fake_seg) for fake_seg in fake_segs]
        if self.opt.things_dis:
            x_fake_things_seg = x_fake_segs[-1].copy()
            x_fake_things_seg["sem_seg"] = torch.zeros_like(x_fake_segs[-1]["sem_seg"])
            x_fake_things_seg["sem_seg"][:, self.opt.things_idx] = x_fake_segs[-1]["sem_seg"][:, self.opt.things_idx]

        if interpolate:
            score = self.netD.interpolate(x_fake_segs[-1], alpha)
            if self.opt.things_dis:
                score_things = self.netD.interpolate(x_fake_things_seg, alpha)
        else:
            score = self.netD(x_fake_segs[-1])
            if self.opt.things_dis:
                score_things = self.netD(x_fake_things_seg)

        loss_gen = self.gan_loss.generator_loss_logits(score).sum()
        loss = loss_gen



        if self.opt.things_dis:
            loss_gen_things = self.gan_loss.generator_loss_logits(score_things).sum()
            loss += self.opt.lambda_adv_things * loss_gen_things

        spread = torch.tensor([])
        fake_sem_mask = torch.tensor([])
        fake_ins_cond = torch.tensor([])
        pseudo_center_mask = torch.tensor([])
        real_sem_cond = torch.tensor([])
        real_ins_cond = torch.tensor([])
        pseudo_ins_center = torch.tensor([])
        pseudo_ins_offset = torch.tensor([])
        entropy = torch.tensor([])
        mentropy = torch.tensor([])
        fake_sem_cond = torch.tensor([])
        fake_center_mask = torch.tensor([])

        loss_sem_entropy = []
        loss_sem_merged_entropy = []
        loss_sem_recover = []
        loss_sem_novelty = []
        loss_ins_recover = []
        loss_pseudo_center = []
        loss_pseudo_offset = []
        loss_sem_spread = []
        loss_sem_conservation = []


        for fake_seg, x_fake_seg in zip(fake_segs, x_fake_segs):

            logprob = torch.log(fake_seg["raw_sem_seg"] + 0.00001)
            entropy = -torch.sum(torch.mul(fake_seg["raw_sem_seg"], logprob), dim=1, keepdim=True)
            loss_sem_entropy.append(torch.mean(entropy))
            # merged entropy
            logprob = torch.log(fake_seg["sem_seg"] + 0.00001)
            mentropy = -torch.sum(torch.mul(fake_seg["sem_seg"], logprob), dim=1, keepdim=True)
            loss_sem_merged_entropy.append(torch.mean(mentropy))

            if self.opt.cond_seg:
                cond_loss = 0


                if self.opt.cond_seg in ["semantic", "panoptic"]:
                    real_sem_cond = real_cond["sem_cond"]
                    fake_sem_cond = torch.mean(fake_seg["raw_sem_seg"], dim=(2, 3))
                    logprob_cond = torch.log(fake_sem_cond + 0.00001)
                    loss_sem_recover.append(F.kl_div(logprob_cond, real_sem_cond, reduction='batchmean'))

                    if 'sem_recover' in self.opt.cond_mode:
                        cond_loss += loss_sem_recover[-1]

                if self.opt.cond_seg in ["instance", "panoptic"] and "density" in self.opt.instance_type:
                    real_ins_cond = real_cond["ins_cond"]
                    fake_ins_cond = torch.sum(fake_seg["ins_density"], dim=(2, 3))
                    loss_ins_recover.append(F.l1_loss(fake_ins_cond, real_ins_cond))

                    if 'ins_recover' in self.opt.cond_mode:
                        cond_loss += loss_ins_recover[-1]

                if 'sem_assisted' in self.opt.cond_mode:
                    fake_sem_mask = fake_seg["sem_mask"]
                    spread = torch.sum(fake_sem_mask, dim=1)
                    loss_sem_spread.append(torch.mean((spread - 1) ** 2))

                    if 'spread' in self.opt.cond_mode:
                        cond_loss += loss_sem_spread[-1] * self.opt.lambda_spread

                if 'entropy' in self.opt.cond_mode:
                    cond_loss += loss_sem_entropy[-1]

                if 'ment' in self.opt.cond_mode:
                    cond_loss += loss_sem_merged_entropy[-1]

                scaled_sem = F.interpolate(sem, size=fake_seg["sem_seg"].size()[2:], mode='nearest')
                # loss_sem_novelty.append(- torch.mean((fake_seg["sem_seg"] - scaled_sem) ** 2))
                eps = 0.00001
                if self.opt.scalnovelty:
                    # scal novelty
                    is_not_bg = torch.argmax(fake_seg["raw_sem_seg"], dim=1) != self.opt.num_semantics
                    fake_codes = fake_seg["sem_seg"].permute(0, 2, 3, 1)[is_not_bg]
                    real_codes = scaled_sem.permute(0, 2, 3, 1)[is_not_bg]
                    scal = torch.sum(fake_codes * real_codes, dim=1)
                    loss_sem_novelty.append(torch.mean(scal))
                else:
                    # invnovelty
                    loss_sem_novelty.append(1 / (torch.mean((fake_seg["sem_seg"] - scaled_sem) ** 2) + eps))
                fake_seg["raw_sem_seg"]
                # loss_sem_novelty.append(1 / (torch.mean((fake_seg["sem_seg"] - scaled_sem) ** 2) + eps))
                loss_sem_conservation.append(torch.mean((fake_seg["sem_seg"] - scaled_sem) ** 2))
                if "novelty" in self.opt.cond_mode:
                    cond_loss += loss_sem_novelty[-1] * self.opt.lambda_novelty
                if 'conservation' in self.opt.cond_mode:
                    cond_loss += loss_sem_conservation[-1]

                loss += cond_loss

            if self.opt.pseudo_supervision:
                with torch.no_grad():
                    pseudo = self.ins_refiner.batch_transform(fake_seg["ins_center"], x_fake_seg["ins_offset"], x_fake_seg["sem_seg"])
                    pseudo_ins_center, pseudo_ins_offset = pseudo
                loss_pseudo_center.append(F.mse_loss(fake_seg["ins_center"], pseudo_ins_center))
                loss_pseudo_offset.append(F.mse_loss(x_fake_seg["ins_offset"], pseudo_ins_offset))
                loss_pseudo = loss_pseudo_center[-1] + loss_pseudo_offset[-1]
                loss += loss_pseudo
        if self.logger:
            # log scalars every step
            self.logger.log_scalar("seg_generator/sem_entropy", loss_sem_entropy, global_iteration)
            self.logger.log_scalar("seg_generator/sem_merged_entropy", loss_sem_merged_entropy, global_iteration)
            self.logger.log_scalar("seg_generator/conservation", loss_sem_conservation, global_iteration)
            self.logger.log_scalar("seg_generator/gen", loss_gen, global_iteration)
            self.logger.log_scalar("seg_generator/novelty", loss_sem_novelty, global_iteration)
            self.logger.log_scalar("seg_generator/gen_things", loss_gen_things, global_iteration)
            self.logger.log_scalar("seg_generator/sem_cond_recover", loss_sem_recover, global_iteration)
            self.logger.log_scalar("seg_generator/sem_ins_recover", loss_ins_recover, global_iteration)
            self.logger.log_scalar("seg_generator/sem_cond_spread", loss_sem_spread, global_iteration)
            self.logger.log_scalar("seg_generator/ins_pseudo_center", loss_pseudo_center, global_iteration)
            self.logger.log_scalar("seg_generator/ins_pseudo_offset", loss_pseudo_offset, global_iteration)
            # log images every few steps
            if log:
                fake_seg = fake_segs[-1]
                x_fake_seg = x_fake_segs[-1]
                with torch.no_grad():
                    raw = fake_seg["raw_sem_seg"].detach().clone()[:16]
                    if not self.opt.fill_crop_only and not self.opt.merged_activation:
                        bg = raw[:, [-1]]
                        raw = raw[:, :-1]
                    fake_raw_sem_seg = torch.zeros_like(sem[:16])
                    sem_label_not_ban = [i for i in range(self.opt.num_semantics) if i not in self.opt.sem_label_ban]
                    fake_raw_sem_seg[:, sem_label_not_ban] = raw
                    if not self.opt.fill_crop_only and not self.opt.merged_activation:
                        fake_raw_sem_seg = torch.cat([fake_raw_sem_seg, bg], dim=1)
                    if fake_seg["ins_center"].size(0) > 0:
                        fake_center_mask = self.ins_refiner.get_peak_mask(fake_seg["ins_center"][:16])
                    if pseudo_ins_center.size(0) > 0 and pseudo_center_mask.size(0) == 0:
                        pseudo_center_mask = self.ins_refiner.get_peak_mask(pseudo_ins_center[:16])
                self.logger.log_semantic_seg("seg_generator/cropped_real", sem[:16].cpu(), 4, global_iteration)
                self.logger.log_semantic_seg("seg_generator/fake", fake_seg["sem_seg"][:16].cpu(), 4, global_iteration)
                self.logger.log_semantic_seg("seg_generator/fake_gumbel", x_fake_seg["sem_seg"][:16].cpu(), 4, global_iteration)
                self.logger.log_cond_distrib("seg_generator/semantic_distrib", real_sem_cond[:16].cpu(), fake_sem_cond[:16].cpu(), 4, 4, global_iteration)
                self.logger.log_img("seg_generator/entropy", entropy[:16].cpu(), 4, global_iteration)
                self.logger.log_img("seg_generator/merged_entropy", mentropy[:16].cpu(), 4, global_iteration)
                self.logger.log_spread("seg_generator/spread", spread[:16].cpu(), 4, global_iteration)
                self.logger.log_semantic_mask("seg_generator/semantic_mask", fake_sem_mask[:1].cpu(), real_sem_cond[:1].cpu(), 16, 4, global_iteration)
                self.logger.log_semantic_seg("seg_generator/fake_raw", fake_raw_sem_seg.cpu(), 4, global_iteration)
                self.logger.log_ins_center("seg_generator/fake_ins_center", fake_seg["ins_center"][:16].cpu(), 4, global_iteration)
                self.logger.log_ins_center("seg_generator/pseudo_ins_center_gumbel", pseudo_ins_center[:16].cpu(), 4, global_iteration)
                self.logger.log_img("seg_generator/fake_center_mask", fake_center_mask[:16].cpu(), 4, global_iteration)
                self.logger.log_img("seg_generator/pseudo_center_mask_gumbel", pseudo_center_mask[:16].cpu(), 4, global_iteration)
                self.logger.log_instance("seg_generator/fake_instance_gumbel", x_fake_seg["sem_seg"][:16].cpu(), fake_center_mask[:16].cpu(), x_fake_seg["ins_offset"][:16].cpu(), 4, global_iteration)
                self.logger.log_instance("seg_generator/pseudo_instance_gumbel", x_fake_seg["sem_seg"][:16].cpu(), pseudo_center_mask[:16].cpu(), pseudo_ins_offset[:16].cpu(), 4, global_iteration)
                self.logger.log_ins_offset("seg_generator/fake_ins_offset_gumbel", x_fake_seg["sem_seg"][:16].cpu(), x_fake_seg["ins_offset"][:16].cpu(), 4, global_iteration)
                self.logger.log_ins_offset("seg_generator/pseudo_ins_offset_gumbel", x_fake_seg["sem_seg"][:16].cpu(), pseudo_ins_offset[:16].cpu(), 4, global_iteration)
                self.logger.log_img("seg_generator/fake_ins_edge", fake_seg["ins_edge"][:16].cpu(), 4, global_iteration)
                self.logger.log_ins_density("seg_generator/fake_ins_density", fake_seg["ins_density"][:16].cpu(), 4, global_iteration)
                self.logger.log_cond_distrib("seg_generator/instance_distrib", real_ins_cond[:16].cpu(), fake_ins_cond[:16].cpu(), 4, 4, global_iteration)

        if hard:
            return loss, x_fake_segs[-1]
        else:
            return loss, fake_segs[-1]


    def compute_discriminator_loss(self, real_seg, fake_seg, interpolate, alpha, log, global_iteration):
        if interpolate:
            real_score = self.netD.interpolate(real_seg, alpha)
            fake_score = self.netD.interpolate(fake_seg, alpha)
            forward = lambda x: self.netD.interpolate(x, alpha)
        else:
            real_score = self.netD(real_seg)
            fake_score = self.netD(fake_seg)
            forward = self.netD

        if self.opt.panoptic:
            real = torch.cat([real_seg["sem_seg"], real_seg["ins_center"], real_seg["ins_offset"], real_seg["ins_edge"],
                              real_seg["ins_density"]], dim=1)
            fake = torch.cat([fake_seg["sem_seg"], fake_seg["ins_center"], fake_seg["ins_offset"], fake_seg["ins_edge"],
                              fake_seg["ins_density"]], dim=1)
        else:
            real = real_seg["sem_seg"]
            fake = fake_seg["sem_seg"]

        loss = self.gan_loss.discriminator_loss_logits(real, fake, real_score, fake_score, forward=forward)

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("seg_generator/dis", loss, global_iteration)
            # log images every few step
            if log:
                real_center_mask = torch.Tensor([])
                if self.opt.panoptic:
                    with torch.no_grad():
                        real_center_mask = self.ins_refiner.get_peak_mask(real_seg["ins_center"])
                self.logger.log_semantic_seg("seg_generator/real", real_seg["sem_seg"][:16].cpu(), 4, global_iteration)
                self.logger.log_ins_center("seg_generator/real_ins_center", real_seg["ins_center"][:16].cpu(), 4, global_iteration)
                self.logger.log_ins_offset("seg_generator/real_ins_offset", real_seg["sem_seg"][:16].cpu(), real_seg["ins_offset"][:16].cpu(), 4, global_iteration)
                self.logger.log_instance("seg_generator/real_instance", real_seg["sem_seg"][:16].cpu(), real_center_mask[:16].cpu(), real_seg["ins_offset"][:16].cpu(), 4, global_iteration)
                self.logger.log_img("seg_generator/real_ins_edge", real_seg["ins_edge"][:16].cpu(), 4, global_iteration)
                self.logger.log_ins_density("seg_generator/real_ins_density", real_seg["ins_density"][:16].cpu(), 4, global_iteration)
        return loss


    def generate_fake(self, real_cond, real_seg, z, interpolate, alpha, hard, log, global_iteration):
        with torch.no_grad():
            sem, real_cond = self.transform_sem_cond(real_seg["sem_seg"].clone(), real_cond)

            if interpolate:
                fake_seg = self.netG.interpolate(z, alpha, sem, cond=real_cond)[-1]
            else:
                fake_seg = self.netG(z, sem, cond=real_cond)[-1]

            x_fake_seg = self.to_discrete(fake_seg)
            fake_sem_cond = torch.mean(fake_seg["raw_sem_seg"], dim=(2, 3))

            if self.opt.cond_seg in ["semantic", "panoptic"]:
                real_sem_cond = real_cond["sem_cond"]
            else:
                real_sem_cond = torch.Tensor([])

        if log and self.logger:
            self.logger.log_semantic_seg("seg_generator/fake", fake_seg["sem_seg"][:16].cpu(), 4, global_iteration)
            self.logger.log_cond_distrib("seg_generator/semantic_distrib", real_sem_cond[:16].cpu(), fake_sem_cond[:16].cpu(), 4, 4, global_iteration)

        if not self.is_train:
            x_fake_seg.update({"real_cropped": sem})
            fake_seg.update({"real_cropped": sem})

        if hard:
            return x_fake_seg
        else:
            return fake_seg


    def to_discrete(self, fake_seg):
        fake_sem_seg = fake_seg["sem_seg"]
        if self.opt.discretization == "gumbel":
            x_fake_sem_seg = self.gumbel_sampler(fake_sem_seg)
        elif self.opt.discretization == "max":
            x_fake_sem_seg = self.max_sampler(fake_sem_seg)
        elif self.opt.discretization == "none":
            x_fake_sem_seg = self.none_sampler(fake_sem_seg)
        else:
            raise ValueError
        fake_ins_center, fake_ins_offset = fake_seg["ins_center"], fake_seg["ins_offset"]
        fake_ins_edge = fake_seg["ins_edge"]
        fake_ins_density = fake_seg["ins_density"]
        fake_raw_sem_seg = fake_seg["raw_sem_seg"]
        x_fake_ins_offset = self.ins_refiner.filter_offset(fake_ins_offset, x_fake_sem_seg)
        x_fake_ins_density = self.ins_refiner.filter_density(fake_ins_density, x_fake_sem_seg)
        x_fake_seg = {"sem_seg": x_fake_sem_seg, "ins_center": fake_ins_center, "ins_offset": x_fake_ins_offset,
                      "ins_edge": fake_ins_edge, "ins_density": x_fake_ins_density, "raw_sem_seg": fake_raw_sem_seg}
        return x_fake_seg


    def max_sampler(self, fake, hard=True, dim=1):
        y_soft = fake
        if hard:
            # straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(fake).scatter_(dim, index, 1.0)
            return (y_hard - y_soft).detach() + y_soft
        else:
            # reparametrization trick.
            return y_soft


    def gumbel_sampler(self, fake, hard=True, dim=1):
        logits = torch.log(fake + 0.00001)
        if torch.isnan(logits.max()).data:
            print(fake.min(), fake.max())

        gumbels = -(torch.empty_like(logits).exponential_()).log()  # ~Gumbel(0, 1)
        gumbels = (logits + gumbels) / self.opt.t  # ~Gumbel(logits, tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            return (y_hard - y_soft).detach() + y_soft
        else:
            # reparametrization trick.
            return y_soft

    def none_sampler(self, fake, hard=True, dim=1):
        return fake