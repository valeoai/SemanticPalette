import torch
import torch.nn.functional as F

from ..models.progressive import ProGANGenerator, ProGANDiscriminator
from ..modules.gan_loss import ImprovedWGANLoss
from ..modules.instance_refiner import InstanceRefiner
from tools.utils import to_cuda
from models import load_network, save_network, print_network


class SegModel(torch.nn.Module):
    def __init__(self, opt, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt = opt
        self.is_main = is_main

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
            g_loss, fake_seg = self.compute_generator_loss(real_cond, z, interpolate, alpha, hard, log, global_iteration)
            fake_seg = self.postprocess_output(fake_seg)
            return g_loss, fake_seg

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(real_cond, real_seg, fake_seg, interpolate, alpha, log, global_iteration)
            return d_loss

        elif mode == 'inference':
            fake_seg = self.generate_fake(real_cond, z, interpolate, alpha, hard, log, global_iteration)
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
        else:
            raise ValueError

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


    def compute_generator_loss(self, real_cond, z, interpolate, alpha, hard, log, global_iteration):

        if interpolate:
            fake_segs = self.netG.interpolate(z, alpha, cond=real_cond)
        else:
            fake_segs = self.netG(z, cond=real_cond)

        if not "inter" in self.opt.cond_mode:
            fake_segs = [fake_segs[-1]]

        x_fake_segs = [self.to_discrete(fake_seg) for fake_seg in fake_segs]

        sem_for_dis = real_cond["sem_cond"] if "original_cgan" in self.opt.cond_mode  else None

        if interpolate:
            score = self.netD.interpolate(x_fake_segs[-1], alpha, sem_cond=sem_for_dis)
        else:
            score = self.netD(x_fake_segs[-1], sem_cond=sem_for_dis)

        loss_gen = self.gan_loss.generator_loss_logits(score).sum()
        loss = loss_gen

        spread = torch.tensor([])
        fake_sem_mask = torch.tensor([])
        fake_ins_cond = torch.tensor([])
        pseudo_center_mask = torch.tensor([])
        fake_raw_filtered_sem_seg = torch.tensor([])
        real_sem_cond = torch.tensor([])
        real_ins_cond = torch.tensor([])
        pseudo_ins_center = torch.tensor([])
        pseudo_ins_offset = torch.tensor([])
        entropy = torch.tensor([])
        fake_sem_cond = torch.tensor([])
        fake_center_mask = torch.tensor([])

        loss_sem_entropy = []
        loss_sem_recover = []
        loss_sem_d_recover = []
        loss_ins_recover = []
        loss_pseudo_center = []
        loss_pseudo_offset = []
        loss_sem_spread = []
        loss_ova = []

        for fake_seg, x_fake_seg in zip(fake_segs, x_fake_segs):
            logprob = torch.log(fake_seg["sem_seg"] + 0.00001)
            entropy = -torch.sum(torch.mul(fake_seg["sem_seg"], logprob), dim=1, keepdim=True)
            loss_sem_entropy.append(torch.mean(entropy))

            if self.opt.cond_seg:
                cond_loss = 0


                if self.opt.cond_seg in ["semantic", "panoptic"]:

                    real_sem_cond = real_cond["sem_cond"]
                    fake_sem_cond = torch.mean(fake_seg["sem_seg"], dim=(2, 3))
                    index = fake_seg["sem_seg"].max(1, keepdim=True)[1]
                    d_fake_sem_seg = torch.zeros_like(fake_seg["sem_seg"]).scatter_(1, index, 1.0)
                    d_fake_sem_cond = torch.mean(d_fake_sem_seg, dim=(2, 3))
                    logprob_cond = torch.log(fake_sem_cond + 0.00001)
                    d_logprob_cond = torch.log(d_fake_sem_cond + 0.00001)
                    loss_sem_recover.append(F.kl_div(logprob_cond, real_sem_cond, reduction='batchmean'))
                    loss_sem_d_recover.append(F.kl_div(d_logprob_cond, real_sem_cond, reduction='batchmean'))

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
                    if len(self.opt.ova_idx) > 0:
                        ova = 0
                        for idx in self.opt.ova_idx:
                            other_idx = [i for i in range(self.opt.num_semantics) if i != idx]
                            ova += torch.mean(torch.sum(fake_sem_mask[:, other_idx], dim=1) * fake_sem_mask[:, idx])
                        loss_ova.append(ova)
                        cond_loss += ova * self.opt.lambda_ova

                    if 'spread' in self.opt.cond_mode:
                        cond_loss += loss_sem_spread[-1]

                if 'entropy' in self.opt.cond_mode:
                    cond_loss += loss_sem_entropy[-1]

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
            self.logger.log_scalar("seg_generator/gen", loss_gen, global_iteration)
            self.logger.log_scalar("seg_generator/sem_cond_recover", loss_sem_recover, global_iteration)
            self.logger.log_scalar("seg_generator/sem_cond_true_recover", loss_sem_d_recover, global_iteration)
            self.logger.log_scalar("seg_generator/sem_ins_recover", loss_ins_recover, global_iteration)
            self.logger.log_scalar("seg_generator/sem_cond_spread", loss_sem_spread, global_iteration)
            self.logger.log_scalar("seg_generator/ins_pseudo_center", loss_pseudo_center, global_iteration)
            self.logger.log_scalar("seg_generator/ins_pseudo_offset", loss_pseudo_offset, global_iteration)
            self.logger.log_scalar("seg_generator/one_versus_all", loss_ova, global_iteration)
            # log images every few steps
            if log:
                fake_seg = fake_segs[-1]
                x_fake_seg = x_fake_segs[-1]
                with torch.no_grad():
                    fake_raw_sem_seg = fake_seg["raw_sem_seg"]
                    if fake_raw_sem_seg.size(0) > 0:
                        fake_raw_filtered_sem_seg = torch.zeros(fake_raw_sem_seg[:16].cpu().shape)
                        fake_raw_filtered_sem_seg[real_sem_cond[:16].cpu()>0] = fake_raw_sem_seg[:16].cpu()[real_sem_cond[:16].cpu()>0]
                    if fake_seg["ins_center"].size(0) > 0:
                        fake_center_mask = self.ins_refiner.get_peak_mask(fake_seg["ins_center"][:16])
                    if pseudo_ins_center.size(0) > 0 and pseudo_center_mask.size(0) == 0:
                        pseudo_center_mask = self.ins_refiner.get_peak_mask(pseudo_ins_center[:16])
                self.logger.log_semantic_seg("seg_generator/fake", fake_seg["sem_seg"][:16].cpu(), 4, global_iteration)
                self.logger.log_semantic_seg("seg_generator/fake_gumbel", x_fake_seg["sem_seg"][:16].cpu(), 4, global_iteration)
                self.logger.log_cond_distrib("seg_generator/semantic_distrib", real_sem_cond[:16].cpu(), fake_sem_cond[:16].cpu(), 4, 4, global_iteration)
                self.logger.log_img("seg_generator/entropy", entropy[:16].cpu(), 4, global_iteration)
                self.logger.log_spread("seg_generator/spread", spread[:16].cpu(), 4, global_iteration)
                self.logger.log_semantic_mask("seg_generator/semantic_mask", fake_sem_mask[:1].cpu(), real_sem_cond[:1].cpu(), 16, 4, global_iteration)
                self.logger.log_semantic_seg("seg_generator/fake_raw", fake_raw_sem_seg[:16].cpu(), 4, global_iteration)
                self.logger.log_semantic_seg("seg_generator/fake_raw_filtered", fake_raw_filtered_sem_seg[:16].cpu(), 4, global_iteration)
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


    def compute_discriminator_loss(self, real_cond, real_seg, fake_seg, interpolate, alpha, log, global_iteration):
        sem_for_dis_real = torch.mean(real_seg["sem_seg"], dim=(2, 3)) if "original_cgan" in self.opt.cond_mode  else None
        sem_for_dis_fake = real_cond["sem_cond"] if "original_cgan" in self.opt.cond_mode  else None

        if interpolate:
            real_score = self.netD.interpolate(real_seg, alpha, sem_cond=sem_for_dis_real)
            fake_score = self.netD.interpolate(fake_seg, alpha, sem_cond=sem_for_dis_fake)
            forward = lambda x: self.netD.interpolate(x, alpha, sem_cond=sem_for_dis_real)
        else:
            real_score = self.netD(real_seg, sem_cond=sem_for_dis_real)
            fake_score = self.netD(fake_seg, sem_cond=sem_for_dis_fake)
            forward = lambda x: self.netD(x, sem_cond=sem_for_dis_real)

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


    def generate_fake(self, real_cond, z, interpolate, alpha, hard, log, global_iteration):
        with torch.no_grad():
            if interpolate:
                fake_seg = self.netG.interpolate(z, alpha, cond=real_cond)[-1]
            else:
                fake_seg = self.netG(z, cond=real_cond)[-1]

            x_fake_seg = self.to_discrete(fake_seg)
            fake_sem_cond = torch.mean(x_fake_seg["sem_seg"], dim=(2, 3))

            if self.opt.cond_seg in ["semantic", "panoptic"]:
                real_sem_cond = real_cond["sem_cond"]
            else:
                real_sem_cond = torch.Tensor([])

        if log and self.logger:
            self.logger.log_semantic_seg("seg_generator/fake", fake_seg["sem_seg"][:16].cpu(), 4, global_iteration)
            self.logger.log_cond_distrib("seg_generator/semantic_distrib", real_sem_cond[:16].cpu(), fake_sem_cond[:16].cpu(), 4, 4, global_iteration)

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
        x_fake_ins_offset = self.ins_refiner.filter_offset(fake_ins_offset, x_fake_sem_seg)
        x_fake_ins_density = self.ins_refiner.filter_density(fake_ins_density, x_fake_sem_seg)
        x_fake_seg = {"sem_seg": x_fake_sem_seg, "ins_center": fake_ins_center, "ins_offset": x_fake_ins_offset,
                      "ins_edge": fake_ins_edge, "ins_density": x_fake_ins_density}
        if self.opt.store_masks:
            x_fake_seg["sem_mask"] = fake_seg["sem_mask"]
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