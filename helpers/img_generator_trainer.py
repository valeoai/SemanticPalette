from apex import amp
import torch
import torch.nn as nn

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urbangan_dir = os.path.dirname(current_dir)
sys.path.insert(0, urbangan_dir)

from tools.options import Options
from tools.engine import Engine
from tools.logger import Logger
from data import create_dataset
from models.img_generator.models.img_model import ImgModel
from models.img_style_generator.models.img_model import ImgModel as ImgStyleModel

class ImgGeneratorTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.old_lr = self.opt.lr

    def step_generator(self, data, log, global_iteration):
        self.opt_g.zero_grad()
        g_loss, fake_img = self.img_model(data, mode='generator', log=log, global_iteration=global_iteration)
        g_loss = self.engine.all_reduce_tensor(g_loss)
        if self.opt.use_amp:
            with amp.scale_loss(g_loss, self.opt_g) as scaled_loss:
                scaled_loss.backward()
        else:
            g_loss.backward()
        self.opt_g.step()
        return fake_img

    def step_discriminator(self, data, log, global_iteration, fake_img=None):
        self.opt_d.zero_grad()
        fake_data = data.copy()
        if fake_img:
            fake_data.update(fake_img)
        d_loss = self.img_model(data, fake_data=fake_data, mode='discriminator', log=log, global_iteration=global_iteration)
        d_loss = self.engine.all_reduce_tensor(d_loss)
        if self.opt.use_amp:
            with amp.scale_loss(d_loss, self.opt_d) as scaled_loss:
                scaled_loss.backward()
        else:
            d_loss.backward()
        self.opt_d.step()

    def step_model(self, data, log, global_iteration):
        fake_img = self.step_generator(data, log, global_iteration)
        self.step_discriminator(data, log, global_iteration, fake_img=fake_img)

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.opt_d.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.opt_g.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            self.dataset = create_dataset(self.opt, load_seg=True, load_img=True)
            self.dataloader, self.datasampler = engine.create_dataloader(self.dataset, self.opt.batch_size, self.opt.num_workers, True)
            is_main = self.opt.local_rank == 0
            logger = Logger(self.opt) if is_main else None
            if self.opt.img_type == "generator":
                self.img_model_on_one_gpu = ImgModel(self.opt, is_train=True, is_main=is_main, logger=logger)
            elif self.opt.img_type == "style_generator":
                self.img_model_on_one_gpu = ImgStyleModel(self.opt, is_train=True, is_main=is_main, logger=logger)
            else:
                raise ValueError
            self.opt_g = self.img_model_on_one_gpu.opt_g
            self.opt_d = self.img_model_on_one_gpu.opt_d
            if self.opt.use_amp:
                optimizer = [self.opt_g, self.opt_d]
                self.img_model_on_one_gpu, optimizer = amp.initialize(self.img_model_on_one_gpu, optimizer,
                                                                      opt_level="O1")
                self.opt_g, self.opt_d = optimizer
            if self.opt.batch_size == 1:
                print("Desactivating batchnorm because batch size is too small")
                self.img_model_on_one_gpu.apply(deactivate_batchnorm)
            self.img_model = engine.data_parallel(self.img_model_on_one_gpu)
            self.img_model.train()

            if self.opt.cont_train:
                start_epoch = self.opt.which_iter
            else:
                start_epoch = 0
            end_epoch = self.opt.niter + self.opt.niter_decay

            global_iteration = start_epoch * int(len(self.dataset) / self.opt.batch_size)

            for epoch in range(start_epoch, end_epoch):
                if self.engine.distributed:
                    self.datasampler.set_epoch(epoch)
                for i, data_i in enumerate(self.dataloader):
                    global_iteration += 1
                    log = (i + 1) % self.opt.log_freq == 0 and is_main
                    if i % self.opt.D_steps_per_G == 0:
                        self.step_model(data_i, log, global_iteration)
                    else:
                        self.step_discriminator(data_i, log, global_iteration)
                    if log:
                        print(f"[Ep{epoch}/{end_epoch}] Iteration {i + 1:05d}/{int(len(self.dataset) / self.opt.batch_size):05d}")

                if self.opt.save_freq > 0 and epoch % self.opt.save_freq == 0 and is_main:
                    self.img_model_on_one_gpu.save_model(epoch, latest=False)
                if self.opt.save_latest_freq > 0 and epoch % self.opt.save_latest_freq == 0 and is_main:
                    self.img_model_on_one_gpu.save_model(epoch, latest=True)

                self.update_learning_rate(epoch)
                print(f"End of epoch, setting lr to {self.old_lr}")

            print('Training was successfully finished.')

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

if __name__ == "__main__":
    opt = Options().parse(load_img_generator=True, save=True)
    ImgGeneratorTrainer(opt["img_generator"]).run()