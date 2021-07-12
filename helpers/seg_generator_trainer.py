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
from data import create_dataset
from models.seg_generator.models.seg_model import SegModel as SegModelGen
from models.seg_completor.models.seg_model import SegModel as SegModelCom


class SegGeneratorTrainer:
    def __init__(self, opt):
        self.opt = opt

    def initialize_training_variables(self):
        if self.opt.batch_size_per_res:
            assert len(self.opt.batch_size_per_res) == int(np.log2(self.opt.max_dim) - 1)
            self.batch_size_per_res = self.opt.batch_size_per_res
        else:
            self.batch_size_per_res = [self.opt.batch_size] * (int(np.log2(self.opt.max_dim)) - 1)

        if self.opt.iter_function_per_res:
            assert len(self.opt.iter_function_per_res) == int(np.log2(self.opt.max_dim) - 1)
            self.iter_function_per_res = [cycle if s == "cycle" else iter for s in self.opt.iter_function_per_res]
        else:
            self.iter_function_per_res = [iter] * (int(np.log2(self.opt.max_dim)) - 1)

        if self.opt.step_mul_per_res:
            assert len(self.opt.step_mul_per_res) == int(np.log2(self.opt.max_dim) - 1)
            step_mul_per_res = self.opt.step_mul_per_res
        else:
            step_mul_per_res = [1] * int(np.log2(self.opt.max_dim) - 1)

        self.steps_per_res = [int(self.opt.niter * len(self.dataset) / self.batch_size_per_res[d]) for d in range(int(np.log2(self.opt.max_dim) - 1))]
        self.steps_per_res = [int(step_mul_per_res[k] * self.steps_per_res[k]) for k in range(len(self.steps_per_res))]

        self.log_step = [int(steps / self.opt.log_per_phase) for steps in self.steps_per_res]

    def set_data_resolution(self, dim):
        dim_ind = int(np.log2(dim)) - 2
        self.dataset.dim_ind = dim_ind
        self.dataloader, self.datasampler = self.engine.create_dataloader(self.dataset,
                                                                          self.batch_size_per_res[dim_ind],
                                                                          self.opt.num_workers, is_train=True)
        self.seg_model_on_one_gpu.ins_refiner.dim_ind = dim_ind
        self.iter_function = self.iter_function_per_res[dim_ind]
        self.loader_iter = self.iter_function(self.dataloader)

    def next_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.epoch += 1
            if self.engine.distributed:
                self.datasampler.set_epoch(self.epoch)
            self.loader_iter = self.iter_function(self.dataloader)
            return next(self.loader_iter)

    def step_model(self, data, dim_ind, phase, iteration, global_iteration, log):
        # manually update cond if cycle iter mode
        if self.opt.estimated_cond and self.iter_function == cycle:
            data["sem_cond"], data["ins_cond"] = self.dataset.cond_estimator.sample(self.engine.batch_size_per_gpu)

        data["z_seg"] = torch.randn(self.engine.batch_size_per_gpu, self.opt.latent_dim)
        alpha = iteration / self.steps_per_res[dim_ind]

        self.opt_g.zero_grad()
        loss_gen, fake_data = self.seg_model(data, interpolate=phase == "fade", alpha=alpha, mode='generator', log=log,
                                             hard=True, global_iteration=global_iteration)
        loss_gen = self.engine.all_reduce_tensor(loss_gen)
        if self.opt.use_amp:
            with amp.scale_loss(loss_gen, self.opt_g) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_gen.backward()
        self.opt_g.step()

        self.opt_d.zero_grad()
        loss_dis = self.seg_model(data, fake_data=fake_data, interpolate=phase == "fade", alpha=alpha,
                                  mode='discriminator', log=log, global_iteration=global_iteration)
        loss_dis = self.engine.all_reduce_tensor(loss_dis)
        if self.opt.use_amp:
            with amp.scale_loss(loss_dis, self.opt_d) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_dis.backward()
        self.opt_d.step()

    def load_checkpoint_information(self):
        steps_per_phase = [k for k in self.steps_per_res for i in range(2)][1:]
        steps = np.cumsum(steps_per_phase)
        res_init, phase_init, iteration_init = None, None, None
        if self.opt.force_res:
            res_init = self.opt.force_res
            iteration_init = 0
            if self.opt.force_phase:
                phase_init = self.opt.force_phase
            else:
                phase_init = "fade"
        else:
            prev_res_step = 0
            for i, step in enumerate(steps):
                if self.opt.which_iter < step:
                    res_step = (i + 1) // 2
                    res_init = int(2 ** (res_step + 2))
                    phase_step = (i + 1) % 2
                    phase_init = ["fade", "stabilize"][phase_step]
                    iteration_init = self.opt.which_iter - prev_res_step
                    break
                prev_res_step = step
        return res_init, phase_init, iteration_init

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            self.dataset = create_dataset(self.opt, load_seg=True, load_img=False)
            self.initialize_training_variables()
            is_main = self.opt.local_rank == 0
            logger = Logger(self.opt) if is_main else None
            if self.opt.seg_type == "generator":
                self.seg_model_on_one_gpu = SegModelGen(self.opt, is_train=True, is_main=is_main, logger=logger)
            elif self.opt.seg_type == "completor":
                self.seg_model_on_one_gpu = SegModelCom(self.opt, is_train=True, is_main=is_main, logger=logger)
            self.opt_g = self.seg_model_on_one_gpu.opt_g
            self.opt_d = self.seg_model_on_one_gpu.opt_d
            if self.opt.use_amp:
                optimizer = [self.opt_g, self.opt_d]
                self.seg_model_on_one_gpu, optimizer = amp.initialize(self.seg_model_on_one_gpu, optimizer,
                                                                      opt_level="O1")
                self.opt_g, self.opt_d = optimizer
            self.seg_model = engine.data_parallel(self.seg_model_on_one_gpu)
            self.seg_model.train()

            if self.opt.cont_train:
                res_init, phase_init, iteration_init = self.load_checkpoint_information()
                global_iteration = self.opt.which_iter
            else:
                res_init, phase_init, iteration_init = 4, "stabilize", 0
                global_iteration = 0
            self.epoch = 0

            self.seg_model_on_one_gpu.netG.res = res_init
            self.seg_model_on_one_gpu.netD.res = res_init
            dim_ind = int(np.log2(res_init)) - 2

            while self.seg_model_on_one_gpu.netG.res <= self.opt.max_dim:
                self.set_data_resolution(self.seg_model_on_one_gpu.netG.res)

                for phase in ["fade", "stabilize"]:
                    if self.seg_model_on_one_gpu.netG.res == 4 and phase == "fade":
                        continue
                    if self.seg_model_on_one_gpu.netG.res == res_init:
                        if phase_init == "stabilize" and phase =="fade":
                            continue

                    if self.seg_model_on_one_gpu.netG.res == res_init and phase_init == phase:
                        start_step = iteration_init
                    else:
                        start_step = 0

                    if is_main:
                        print(f"Training at resolution {self.seg_model_on_one_gpu.netG.res} "
                              f"with phase {phase} "
                              f"and batch size {self.batch_size_per_res[dim_ind]}")

                    for i in np.arange(start_step, self.steps_per_res[dim_ind]):
                        data_i = self.next_batch()
                        log = (i + 1) % self.log_step[dim_ind] == 0 and is_main
                        self.step_model(data_i, dim_ind, phase, i, global_iteration, log)
                        global_iteration += 1

                        if log:
                            print(f"Res {self.seg_model_on_one_gpu.netG.res:03d}, {phase.rjust(9)}, "
                                  f"# labels{self.opt.num_semantics:03d}: "
                                  f"Iteration {i + 1:05d}/{self.steps_per_res[dim_ind]:05d}")

                        if self.opt.save_freq > 0 and global_iteration % self.opt.save_freq == 0 and is_main:
                            self.seg_model_on_one_gpu.save_model(global_iteration, latest=False)
                        if self.opt.save_latest_freq > 0 and global_iteration % self.opt.save_latest_freq == 0 and is_main:
                            self.seg_model_on_one_gpu.save_model(global_iteration, latest=True)
                    if phase == "stabilize" and self.opt.save_at_every_res and is_main:
                        self.seg_model_on_one_gpu.save_model(self.seg_model_on_one_gpu.netG.res, latest=False)
                self.seg_model_on_one_gpu.netG.res *= 2
                self.seg_model_on_one_gpu.netD.res *= 2
                dim_ind += 1

            print('Training was successfully finished.')


if __name__ == "__main__":
    opt = Options().parse(load_seg_generator=True, save=True)
    SegGeneratorTrainer(opt["seg_generator"]).run()
