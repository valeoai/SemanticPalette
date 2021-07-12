import torch

from models import load_network, save_network
import models.img_generator.models.networks as networks


class ImgModel(torch.nn.Module):
    def __init__(self, opt, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt = opt
        self.is_main = is_main

        self.netG, self.netD, self.netE, self.netD2 = self.initialize_networks(is_train)

        if is_train:
            self.opt_g, self.opt_d = self.create_optimizers(self.opt)
            tensor = torch.cuda.HalfTensor if opt.use_amp else torch.cuda.FloatTensor
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=tensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

        self.logger = logger if self.is_main else None


    def forward(self, data, fake_data={}, tgt_data=None, mode="", log=False, global_iteration=0, suffix="", has_gt=True, dis="both"):
        real_seg, real_image = self.preprocess_input(data)
        tgt_image = self.preprocess_input(tgt_data)[1] if tgt_data is not None else real_image
        fake_seg, fake_image = self.preprocess_input(fake_data, is_fake=True)

        if mode == 'generator':
            g_loss, fake_img = self.compute_generator_loss(real_seg, real_image, tgt_image, log, global_iteration, suffix, has_gt, dis)
            return g_loss, {"img": fake_img}
        
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(real_seg, real_image, tgt_image, fake_seg, fake_image, log, global_iteration, suffix, has_gt, dis)
            return d_loss
        
        elif mode == 'inference':
            with torch.no_grad():
                fake_img, _ = self.generate_fake(real_seg, real_image, log=log, global_iteration=global_iteration)
            return {"img": fake_img}
        
        else:
            raise ValueError(f"mode '{mode}' is invalid")


    def create_optimizers(self, opt):
        g_params = list(self.netG.parameters())
        if opt.use_vae:
            g_params += list(self.netE.parameters())
        d_params = list(self.netD.parameters())
        if opt.use_d2:
            d_params += list(self.netD2.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            g_lr, d_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            g_lr, d_lr = opt.lr / 2, opt.lr * 2

        optimizer_g = torch.optim.Adam(g_params, lr=g_lr, betas=(beta1, beta2))
        optimizer_d = torch.optim.Adam(d_params, lr=d_lr, betas=(beta1, beta2))

        return optimizer_g, optimizer_d


    def save_model(self, global_iteration, latest):
        save_network(self.netG, "img_g", global_iteration, self.opt, latest=latest)
        save_network(self.netD, "img_d", global_iteration, self.opt, latest=latest)
        if self.opt.use_vae:
            save_network(self.netE, "img_e", global_iteration, self.opt, latest=latest)
        if self.opt.use_d2:
            save_network(self.netD2, "img_d2", global_iteration, self.opt, latest=latest)


    def save_d2(self, global_iteration, latest):
        save_network(self.netD2, "img_d2", global_iteration, self.opt, latest=latest)


    def initialize_networks(self, is_train=True):
        netG = networks.define_G(self.opt)
        netD = networks.define_D(self.opt) if is_train else None
        netE = networks.define_E(self.opt) if self.opt.use_vae else None
        netD2 = networks.define_D2(self.opt) if is_train and self.opt.use_d2 else None

        if self.is_main:
            netG = load_network(netG, "img_g", self.opt)
            if is_train:
                netD = load_network(netD, "img_d", self.opt)
                if self.opt.use_d2:
                    netD2 = load_network(netD2, "img_d2", self.opt,
                                         override_iter=self.opt.which_iter_d2,
                                         override_load_path=self.opt.load_path_d2)
            if self.opt.use_vae:
                netE = load_network(netE, "img_e", self.opt)

        return netG, netD, netE, netD2


    def preprocess_input(self, data, is_fake=False):
        data['sem_seg'] = data['sem_seg'].cuda() if 'sem_seg' in data else torch.tensor([])
        data['ins_center'] = data['ins_center'].cuda() if 'ins_center' in data else torch.tensor([])
        data['ins_offset'] = data['ins_offset'].cuda() if 'ins_offset' in data else torch.tensor([])
        data['ins_edge'] = data['ins_edge'].cuda() if 'ins_edge' in data else torch.tensor([])
        data['img'] = data['img'].cuda() if 'img' in data else torch.tensor([])

        if is_fake:
            data['sem_seg'] = data['sem_seg'].detach()
            data['ins_center'] = data['ins_center'].detach()
            data['ins_offset'] = data['ins_offset'].detach()
            data['ins_edge'] = data['ins_edge'].detach()
            data['img'] = data['img'].detach()

        if self.opt.panoptic and data['sem_seg'].size(0) > 0:
            if self.opt.instance_type_for_img == 'center_offset':
                flat_center = torch.max(data['ins_center'], dim=1, keepdim=True)[0]
                seg = torch.cat((data['sem_seg'], flat_center, data['ins_offset']), dim=1)
            elif 'edge' in self.opt.instance_type_for_img:
                seg = torch.cat((data['sem_seg'], data['ins_edge']), dim=1)
            else:
                raise ValueError
        else:
            seg = data['sem_seg']

        img = data['img']

        return seg, img


    def compute_generator_loss(self, real_seg, real_image, tgt_image, log, global_iteration, suffix, has_gt, dis):
        is_real = suffix != "_from_fake"
        g_loss = 0
        gan_feat_loss = None
        vgg_loss = None
        fake_image, kld_loss = self.generate_fake(real_seg, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            g_loss += kld_loss

        gan_loss_r = None
        gan_loss_t = None

        if dis == "d" or dis == "both":
            img_for_d = self.opt.img_for_d_real if is_real else self.opt.img_for_d_fake
            if img_for_d == "source" or img_for_d == "both":
                sem_only = self.opt.sem_only_real and is_real
                seg = real_seg if has_gt else None
                feat_fake, pred_fake, feat_real, pred_real = self.discriminate(seg, seg, fake_image, real_image, sem_only=sem_only)
                gan_loss_r = self.criterionGAN(pred_fake, True, for_discriminator=False)
                g_loss += gan_loss_r
            if img_for_d == "target" or img_for_d == "both":
                _, pred_fake, _, _ = self.discriminate(None, None, fake_image, tgt_image)
                gan_loss_t = self.criterionGAN(pred_fake, True, for_discriminator=False)
                g_loss += gan_loss_t

        gan_loss_d2_r = None
        gan_loss_d2_t = None

        if self.opt.use_d2 and (dis == "d2" or dis == "both"):
            img_for_d2 = self.opt.img_for_d2_real if is_real else self.opt.img_for_d2_fake
            alpha = 1. if suffix != "_from_real" else self.opt.lambda_d2_from_real
            if img_for_d2 == "source" or img_for_d2 == "both":
                pred_fake_d2, _ = self.discriminate2(fake_image, real_image)
                gan_loss_d2_r = self.criterionGAN(pred_fake_d2, True, for_discriminator=False)
                g_loss += gan_loss_d2_r * self.opt.lambda_d2 * alpha
            if img_for_d2 == "target" or img_for_d2 == "both":
                pred_fake_d2, _ = self.discriminate2(fake_image, tgt_image)
                gan_loss_d2_t = self.criterionGAN(pred_fake_d2, True, for_discriminator=False)
                g_loss += gan_loss_d2_t * self.opt.lambda_d2 * alpha

        if not self.opt.no_ganFeat_loss and has_gt:
            num_D = len(feat_fake)
            gan_feat_loss = torch.cuda.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                if self.opt.netD == "fpse":
                    unweighted_loss = self.criterionFeat(feat_fake[i], feat_real[i].detach())
                    gan_feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                else:
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(feat_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(feat_fake[i][j], feat_real[i][j].detach())
                        gan_feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            g_loss += gan_feat_loss

        if not self.opt.no_vgg_loss and has_gt:
            vgg_loss = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
            g_loss += vgg_loss

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("img_generator/kld" + suffix, kld_loss, global_iteration)
            self.logger.log_scalar("img_generator/gan" + suffix, gan_loss_r, global_iteration)
            self.logger.log_scalar("img_generator/gan_t" + suffix, gan_loss_t, global_iteration)
            self.logger.log_scalar("img_generator/gan2" + suffix, gan_loss_d2_r, global_iteration)
            self.logger.log_scalar("img_generator/gan2_t" + suffix, gan_loss_d2_t, global_iteration)
            self.logger.log_scalar("img_generator/gan_feat" + suffix, gan_feat_loss, global_iteration)
            self.logger.log_scalar("img_generator/vgg" + suffix, vgg_loss, global_iteration)
            # log images every few steps
            if log:
                self.logger.log_img("img_generator/fake" + suffix, fake_image[:16].cpu(), 4, global_iteration, normalize=True, range=(-1, 1))
                self.logger.log_semantic_seg("img_generator/sem_seg" + suffix, real_seg[:16, :self.opt.num_semantics].cpu(), 4, global_iteration)

        return g_loss, fake_image


    def compute_discriminator_loss(self, real_seg, real_image, tgt_image, fake_seg, fake_image, log, global_iteration, suffix, has_gt, dis):
        is_real = suffix != "_from_fake"
        d_loss = 0

        if fake_image.size(0) == 0:
            with torch.no_grad():
                fake_image, _ = self.generate_fake(real_seg, real_image)
                fake_image = fake_image.detach()
                fake_image.requires_grad_()

        d_loss_fake_r = None
        d_loss_real_r = None
        d_loss_fake_t = None
        d_loss_real_t = None

        if dis == "d" or dis == "both":
            img_for_d = self.opt.img_for_d_real if is_real else self.opt.img_for_d_fake
            if img_for_d == "source" or img_for_d == "both":
                sem_only = self.opt.sem_only_real and is_real
                seg_r = real_seg if has_gt else None
                seg_f = fake_seg if has_gt else None
                _, pred_fake, _, pred_real = self.discriminate(seg_f, seg_r, fake_image, real_image, sem_only=sem_only)
                d_loss_fake_r = self.criterionGAN(pred_fake, False, for_discriminator=True)
                d_loss_real_r = self.criterionGAN(pred_real, True, for_discriminator=True)
                d_loss += d_loss_fake_r + d_loss_real_r
            if img_for_d == "target" or img_for_d == "both":
                _, pred_fake, _, pred_real = self.discriminate(None, None, fake_image, tgt_image)
                d_loss_fake_t = self.criterionGAN(pred_fake, False, for_discriminator=True)
                d_loss_real_t = self.criterionGAN(pred_real, True, for_discriminator=True)
                d_loss += d_loss_fake_t + d_loss_real_t

        d_loss_fake_d2_r = None
        d_loss_real_d2_r = None
        d_loss_fake_d2_t = None
        d_loss_real_d2_t = None

        if self.opt.use_d2 and (dis == "d2" or dis == "both"):
            img_for_d2 = self.opt.img_for_d2_real if is_real else self.opt.img_for_d2_fake
            alpha = 1. if suffix != "_from_real" else self.opt.lambda_d2_from_real
            if img_for_d2 == "source" or img_for_d2 == "both":
                pred_fake_d2, pred_real_d2 = self.discriminate2(fake_image, real_image)
                d_loss_fake_d2_r = self.criterionGAN(pred_fake_d2, False, for_discriminator=True)
                d_loss_real_d2_r = self.criterionGAN(pred_real_d2, True, for_discriminator=True)
                d_loss_d2 = d_loss_fake_d2_r + d_loss_real_d2_r
            if img_for_d2 == "target" or img_for_d2 == "both":
                pred_fake_d2, pred_real_d2 = self.discriminate2(fake_image, tgt_image)
                d_loss_fake_d2_t = self.criterionGAN(pred_fake_d2, False, for_discriminator=True)
                d_loss_real_d2_t = self.criterionGAN(pred_real_d2, True, for_discriminator=True)
                d_loss_d2 = d_loss_fake_d2_t + d_loss_real_d2_t
            d_loss += d_loss_d2 * self.opt.lambda_d2 * alpha

        if self.logger:
            # log scalars every step
            self.logger.log_scalar("img_generator/dis_fake" + suffix, d_loss_fake_r, global_iteration)
            self.logger.log_scalar("img_generator/dis_real" + suffix, d_loss_real_r, global_iteration)
            self.logger.log_scalar("img_generator/dis_fake_t" + suffix, d_loss_fake_t, global_iteration)
            self.logger.log_scalar("img_generator/dis_real_t" + suffix, d_loss_real_t, global_iteration)
            self.logger.log_scalar("img_generator/dis_fake_d2" + suffix, d_loss_fake_d2_r, global_iteration)
            self.logger.log_scalar("img_generator/dis_real_d2" + suffix, d_loss_real_d2_r, global_iteration)
            self.logger.log_scalar("img_generator/dis_fake_d2_t" + suffix, d_loss_fake_d2_t, global_iteration)
            self.logger.log_scalar("img_generator/dis_real_d2_t" + suffix, d_loss_real_d2_t, global_iteration)
            # log images every few steps
            if log:
                self.logger.log_img("img_generator/real", real_image[:16].cpu(), 4, global_iteration, normalize=True, range=(-1, 1))
                self.logger.log_img("img_generator/target", tgt_image[:16].cpu(), 4, global_iteration, normalize=True, range=(-1, 1))

        return d_loss


    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def generate_fake(self, real_seg, real_image=torch.tensor([]), compute_kld_loss=False, log=False, global_iteration=0):
        z = None
        kld_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                kld_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(real_seg, z=z)

        if log:
            self.logger.log_img("img_generator/real", real_image[:16].cpu(), 4, global_iteration, normalize=True, range=(-1, 1))
            self.logger.log_semantic_seg("img_generator/sem_seg", real_seg[:16, :self.opt.num_semantics].cpu(), 4, global_iteration)
            self.logger.log_img("img_generator/fake", fake_image[:16].cpu(), 4, global_iteration, normalize=True, range=(-1, 1))

        assert (not compute_kld_loss) or self.opt.use_vae, "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, kld_loss


    def discriminate(self, fake_seg, real_seg, fake_image, real_image, sem_only=False):
        if self.opt.netD == "fpse":
            fake_and_real_image = torch.cat([fake_image, real_image], dim=0)
            fake_and_real_seg = torch.cat([fake_seg, real_seg], dim=0) if fake_seg is not None else None
            discriminator_out = self.netD(fake_and_real_image, segmap=fake_and_real_seg, sem_alignment_only=sem_only)
            (feat_fake, pred_fake), (feat_real, pred_real) = self.divide_pred(discriminator_out)
        else:
            fake_concat = torch.cat([fake_seg, fake_image], dim=1)
            real_concat = torch.cat([real_seg, real_image], dim=1)
            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
            discriminator_out = self.netD(fake_and_real)
            pred_fake, pred_real = self.divide_pred(discriminator_out)
            feat_fake, feat_real = pred_fake, pred_real
        return feat_fake, pred_fake, feat_real, pred_real


    def discriminate2(self, fake_image, real_image):
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        discriminator_out = self.netD2(fake_and_real)
        if self.opt.netD == "fpse":
            discriminator_out = discriminator_out[1]
        pred_fake_d2, pred_real_d2 = self.divide_pred(discriminator_out)
        return pred_fake_d2, pred_real_d2


    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu
