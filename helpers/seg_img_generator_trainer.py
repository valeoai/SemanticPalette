from apex import amp
import torch
import numpy as np
from itertools import cycle

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urbangan_dir = os.path.dirname(current_dir)
sys.path.insert(0, urbangan_dir)

from tools.options import Options
from tools.engine import Engine
from tools.logger import Logger
from tools.utils import dict_to_cpu
from tools.fid.inception import InceptionV3
from tools.fid.fid_score import calculate_fid_given_acts, get_activations
from data import create_dataset, create_dataloader
from models.seg_img_model import SegImgModel


class SegImgGeneratorTrainer:
    def __init__(self, opt, is_train=True):
        self.img_opt = opt["img_generator"]
        self.seg_opt = opt["seg_generator"]
        self.tgt_dataset_opt = opt["extra_dataset"]
        self.opt = opt["base"]
        self.old_img_lr = self.img_opt.lr
        self.iter_function = cycle if self.opt.iter_function == "cycle" else iter
        self.has_tgt_dataset = self.tgt_dataset_opt.dataset is not None
        if self.has_tgt_dataset:
            print(f"Target images are loaded from [{self.tgt_dataset_opt.dataset}] dataset")

    def step_model(self, data, log, global_iteration):
        data["z_seg"] = torch.randn(self.engine.batch_size_per_gpu, self.seg_opt.latent_dim)
        tgt_data = self.next_tgt_batch()
        tgt_data = tgt_data if tgt_data is not None else data
        self.opt_g_seg.zero_grad()
        self.opt_g_img.zero_grad()
        g_loss, fake = self.seg_img_model(data, tgt_data, mode='generator', log=log, global_iteration=global_iteration)
        g_loss = self.engine.all_reduce_tensor(g_loss)
        if self.opt.use_amp:
            with amp.scale_loss(g_loss, [self.opt_g_seg, self.opt_g_img]) as scaled_loss:
                scaled_loss.backward()
        else:
            g_loss.backward()
        self.opt_g_seg.step()
        self.opt_g_img.step()

        self.opt_d_seg.zero_grad()
        self.opt_d_img.zero_grad()
        d_loss = self.seg_img_model(data, tgt_data, fake=fake, mode='discriminator', log=log, global_iteration=global_iteration)
        d_loss = self.engine.all_reduce_tensor(d_loss)
        if self.opt.use_amp:
            with amp.scale_loss(d_loss, [self.opt_d_seg, self.opt_d_img]) as scaled_loss:
                scaled_loss.backward()
        else:
            d_loss.backward()
        self.opt_d_seg.step()
        self.opt_d_img.step()

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.img_opt.lr / self.opt.niter_decay
            new_lr = self.old_img_lr - lrd
        else:
            new_lr = self.old_img_lr

        if new_lr != self.old_img_lr:
            if self.img_opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.seg_img_model_on_one_gpu.img_model.opt_d.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.seg_img_model_on_one_gpu.img_model.opt_g.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_img_lr, new_lr))
            self.old_img_lr = new_lr

    def compute_fid(self, real_fake='fake'):
        dims = 2048
        nums_fid = self.opt.nums_fid
        batchsize = self.opt.eval_batchsize
        all_reals = np.zeros((int(nums_fid / batchsize) * batchsize, dims))
        all_fakes = np.zeros((int(nums_fid / batchsize) * batchsize, dims))

        for i in range(int(nums_fid / batchsize)):
            data = self.next_valid_batch()
            data["img"] = data["img"].cuda()

            with torch.no_grad():
                real_acts = get_activations(data["img"], self.inception_model, batchsize, cuda=True)
            all_reals[i * batchsize:i * batchsize + real_acts.shape[0], :] = real_acts

            if real_fake == 'fake':
                with torch.no_grad():
                    data["z_seg"] = torch.randn(batchsize, self.seg_opt.latent_dim)
                    fake_seg = self.seg_img_model_on_one_gpu.seg_model(data, mode="inference", hard=True)
                    fake_img_f = self.seg_img_model_on_one_gpu.img_model(fake_seg, mode="inference")
                    fake_acts = get_activations(fake_img_f["img"], self.inception_model, batchsize, cuda=True)
                all_fakes[i * batchsize:i * batchsize + fake_acts.shape[0], :] = fake_acts
            else:
                with torch.no_grad():
                    fake_img_r = self.seg_img_model_on_one_gpu.img_model(data, mode="inference")
                    fake_acts = get_activations(fake_img_r["img"], self.inception_model, batchsize, cuda=True)
                all_fakes[i * batchsize:i * batchsize + fake_acts.shape[0], :] = fake_acts

        fid_eval = calculate_fid_given_acts(all_reals, all_fakes)
        return fid_eval

    def next_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.loader_iter = self.iter_function(self.dataloader)
            return next(self.loader_iter)

    def next_valid_batch(self):
        try:
            return next(self.val_loader_iter)
        except StopIteration:
            self.val_loader_iter = iter(self.valid_dataloader)
            return next(self.val_loader_iter)

    def next_tgt_batch(self):
        if self.has_tgt_dataset:
            try:
                return next(self.tgt_loader_iter)
            except StopIteration:
                self.tgt_loader_iter = iter(self.tgt_dataloader)
                return next(self.tgt_loader_iter)
        else:
            return None

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            self.dataset = create_dataset(self.opt, load_seg=True, load_img=True)
            self.dataloader, self.datasampler = engine.create_dataloader(self.dataset, self.opt.batch_size, self.opt.num_workers, True)
            if self.has_tgt_dataset:
                self.tgt_dataset = create_dataset(self.tgt_dataset_opt, load_img=True)
                self.tgt_dataloader, self.tgt_datasampler = engine.create_dataloader(self.tgt_dataset, self.opt.batch_size, self.opt.num_workers, True)
                self.tgt_loader_iter = iter(self.tgt_dataloader)
            is_main = self.opt.local_rank == 0
            logger = Logger(self.opt) if is_main else None
            self.seg_img_model_on_one_gpu = SegImgModel(self.seg_opt, self.img_opt, is_train=True, is_main=is_main, logger=logger)
            self.opt_g_seg = self.seg_img_model_on_one_gpu.seg_model.opt_g
            self.opt_g_img = self.seg_img_model_on_one_gpu.img_model.opt_g
            self.opt_d_seg = self.seg_img_model_on_one_gpu.seg_model.opt_d
            self.opt_d_img = self.seg_img_model_on_one_gpu.img_model.opt_d
            if self.opt.use_amp:
                optimizer = [self.opt_g_seg, self.opt_g_img, self.opt_d_seg, self.opt_d_img]
                self.seg_img_model_on_one_gpu, optimizer = amp.initialize(self.seg_img_model_on_one_gpu, optimizer,
                                                                          opt_level="O1")
                self.opt_g_seg, self.opt_g_img, self.opt_d_seg, self.opt_d_img = optimizer

            self.seg_img_model = engine.data_parallel(self.seg_img_model_on_one_gpu)
            self.seg_img_model.train()

            if is_main and not self.opt.no_eval:
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                self.inception_model = InceptionV3([block_idx])
                self.inception_model.cuda()
                eval_opt = self.opt if self.opt.eval_dataset == "base" else self.tgt_dataset_opt
                self.valid_dataset = create_dataset(eval_opt, load_seg=True, load_img=True, phase='valid')
                self.valid_dataloader = create_dataloader(self.valid_dataset, self.opt.eval_batchsize,
                                                          self.opt.num_workers, is_train=True) # set to train to drop last
                self.val_loader_iter = iter(self.valid_dataloader)


            if self.seg_opt.cont_train and self.img_opt.cont_train:
                assert self.seg_opt.which_iter == self.img_opt.which_iter
                start_epoch = self.seg_opt.which_iter
            else:
                start_epoch = 0

            global_iteration = start_epoch * int(len(self.dataset) / self.opt.batch_size)
            end_epoch = self.opt.niter + self.opt.niter_decay
            self.loader_iter = self.iter_function(self.dataloader)

            for epoch in range(end_epoch):
                if self.engine.distributed:
                    self.datasampler.set_epoch(epoch)
                    if self.has_tgt_dataset:
                        self.tgt_datasampler.set_epoch(epoch)

                if epoch % self.opt.eval_freq == 0 and is_main and not self.opt.no_eval:
                    self.seg_img_model_on_one_gpu.eval()
                    fid = self.compute_fid(real_fake='fake')
                    logger.log_scalar("fid_fake", fid, global_iteration)
                    fid = self.compute_fid(real_fake='real')
                    logger.log_scalar("fid_real", fid, global_iteration)
                    self.seg_img_model_on_one_gpu.train()

                for i in range(len(self.dataloader)):
                    data_i = self.next_batch()
                    global_iteration += 1
                    log = global_iteration % self.opt.log_freq == 0 and is_main
                    self.step_model(data_i, log, global_iteration)
                    if self.opt.iter_function == "cycle":
                        dict_to_cpu(data_i)
                    if log:
                        print(f"[Ep{epoch}/{end_epoch}] Iteration {i + 1:05d}/{int(len(self.dataset) / self.opt.batch_size):05d}")

                if self.opt.save_freq > 0 and epoch % self.opt.save_freq == 0 and is_main:
                    self.seg_img_model_on_one_gpu.save_model(epoch, latest=False)
                if self.opt.save_latest_freq > 0 and epoch % self.opt.save_latest_freq == 0 and is_main:
                    self.seg_img_model_on_one_gpu.save_model(epoch, latest=True)

                self.update_learning_rate(epoch)
                if is_main:
                    print(f"End of epoch, setting img lr to {self.old_img_lr}")

            print('Training was successfully finished.')

if __name__ == "__main__":
    opt = Options().parse(load_seg_generator=True, load_img_generator=True, load_extra_dataset=True, save=True)
    SegImgGeneratorTrainer(opt).run()