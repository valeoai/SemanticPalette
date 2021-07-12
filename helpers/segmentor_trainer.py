import torch
from apex import amp
from math import ceil
import random
import PIL
from tqdm import tqdm
#torch.multiprocessing.set_start_method('spawn', force=True)

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urbangan_dir = os.path.dirname(current_dir)
sys.path.insert(0, urbangan_dir)

from tools.options import Options
from tools.engine import Engine
from tools.logger import Logger
from tools.utils import get_confusion_matrix
from models.segmentor.models.segmentor import Segmentor
from models.segmentor_plus.models.segmentor import Segmentor as SegmentorPlus
from inplace_abn import InPlaceABN, InPlaceABNSync
from data.base_dataset import get_transform
#torch.autograd.set_detect_anomaly(True)

class SegmentorTrainer:
    def __init__(self, opt):
        self.opt = opt["segmentor"]
        self.tgt_dataset_opt = opt["extra_dataset"]
        self.all_opt = opt
        if self.opt.advent or self.opt.duo:
            assert self.tgt_dataset_opt.dataset is not None
            print(f"Target pairs for duo/advent training are loaded from [{self.tgt_dataset_opt.dataset}] dataset")

    def step_model(self, data, global_iteration, log):
        # try:
        # import pdb; pdb.set_trace()
        if self.opt.advent or self.opt.duo:
            tgt_data = self.next_tgt_batch()

        self.opt_s.zero_grad()
        if self.opt.advent:
            loss, pred_data, delta = self.segmentor(data, tgt_data, global_iteration=global_iteration, log=log, mode='segmentor_advent', hard=False)
        else:
            loss, _, delta = self.segmentor(data, global_iteration=global_iteration, log=log, mode='segmentor', hard=False)
        loss = self.engine.all_reduce_tensor(loss)
        if self.opt.use_amp:
            with amp.scale_loss(loss, self.opt_s) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.opt.duo:
            loss, _, _ = self.segmentor(tgt_data, global_iteration=global_iteration, log=log, mode='segmentor', hard=False, suffix="_for_duo")
            loss = self.engine.all_reduce_tensor(loss)
            if self.opt.use_amp:
                with amp.scale_loss(loss, self.opt_s) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        self.opt_s.step()


        if self.opt.advent:
            self.opt_d.zero_grad()
            loss = self.segmentor({}, pred_data=pred_data, global_iteration=global_iteration, log=log, mode='discriminator_advent')
            loss = self.engine.all_reduce_tensor(loss)
            if self.opt.use_amp:
                with amp.scale_loss(loss, self.opt_d) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.opt_d.step()

        # record delta for synthetic dataset sampler
        if self.opt.synthetic_dataset and not self.opt.semi:
            self.train_dataset.set_batch_delta(delta)

        # print("allocated", torch.cuda.memory_allocated())
        # print("cached", torch.cuda.memory_cached())
        # torch.cuda.empty_cache()
        # import pdb; pdb.set_trace()
        # except:
        #     print("error at iteration", global_iteration)
        #     import pdb; pdb.set_trace()


    def slide_pred(self, data, window_size):
        img = data["img"]
        b, c, h, w = img.shape
        pred_sem_seg = torch.zeros((b, self.opt.num_semantics, h, w)).cuda()
        pred_sem_count = torch.zeros((h, w)).cuda()
        min_overlap = 1 / 3
        win_h, win_w = window_size
        win_rows = int(ceil((h - win_h) / (win_h * (1 - min_overlap)))) + 1
        win_cols = int(ceil((w - win_w) / (win_w * (1 - min_overlap)))) + 1
        overlap_h = 1 - (h - win_h) / (win_h * (win_rows - 1)) if win_rows > 1 else 0
        overlap_w = 1 - (w - win_w) / (win_w * (win_cols - 1)) if win_cols > 1 else 0
        stride_h = (1 - overlap_h) * win_h
        stride_w = (1 - overlap_w) * win_w
        for row in range(win_rows):
            for col in range(win_cols):
                x1 = int(col * stride_w)
                y1 = int(row * stride_h)
                x2 = x1 + win_w
                y2 = y1 + win_h
                slide_data = {"img": img[:, :, y1:y2, x1:x2]}
                pred = self.segmentor(slide_data, mode='inference', hard=False)
                pred_sem_seg[:, :, y1:y2, x1:x2] = pred["sem_seg"]
                pred_sem_count[y1:y2, x1:x2] += 1
        pred_sem_seg /= pred_sem_count
        return {"sem_seg": pred_sem_seg}

    def eval_model(self, data, global_iteration, log):
        if self.opt.slide_eval:
            pred_seg = self.slide_pred(data, window_size=self.opt.fixed_crop)
            if self.opt.multi_scale_eval:
                if self.opt.aspect_ratio > 1:
                    size = [self.opt.fixed_crop[0], int(self.opt.fixed_crop[0] * self.opt.aspect_ratio)]
                else:
                    size = [int(self.opt.fixed_crop[1] / self.opt.aspect_ratio), self.opt.fixed_crop[1]]
                small_data = {"img": torch.nn.functional.interpolate(data["img"], size=size, mode="bilinear")}
                pred_small_seg = self.slide_pred(small_data, window_size=self.opt.fixed_crop)
                resized_seg = torch.nn.functional.interpolate(pred_small_seg["sem_seg"], size=data["img"].shape[-2:], mode="bilinear")
                pred_seg["sem_seg"] = (pred_seg["sem_seg"] + resized_seg) / 2
        else:
            pred_seg = self.segmentor(data, global_iteration=global_iteration, mode='inference', hard=False)
        pred_sem_seg = pred_seg["sem_seg"]
        sem_index_pred = pred_sem_seg.max(dim=1, keepdim=True)[1]
        if "flat_seg" in data and not 0 in data["flat_seg"].size():
            real_sem_seg = data["flat_seg"].unsqueeze(1).cuda()
            sem_index_real = data["flat_seg"].unsqueeze(1).cuda()
        else:
            real_sem_seg = data["sem_seg"].cuda()
            sem_index_real = real_sem_seg.max(dim=1, keepdim=True)[1]
        if log:
            self.segmentor_on_one_gpu.logger.log_img("segmentor/val/img", data["img"][:16].cpu(), 4, global_iteration, normalize=True, range=(-1, 1))
            self.segmentor_on_one_gpu.logger.log_semantic_seg("segmentor/val/real", real_sem_seg[:16].cpu(), 4, global_iteration)
            self.segmentor_on_one_gpu.logger.log_semantic_seg("segmentor/val/pred", pred_sem_seg[:16].cpu(), 4, global_iteration)
        confusion_matrix = get_confusion_matrix(sem_index_real, sem_index_pred, self.opt.num_semantics)
        return confusion_matrix

    def update_learning_rate(self, global_iteration, min_lr=1e-6):
        total_iterations = int(self.opt.niter * self.dataset_size / self.opt.batch_size)
        lr = max(self.opt.lr * ((1 - float(global_iteration) / total_iterations) ** (self.opt.power)), min_lr)
        if self.opt.plus:
            self.opt_s.param_groups[0]['lr'] = 0.1 * lr
            self.opt_s.param_groups[1]['lr'] = lr
        else:
            self.opt_s.param_groups[0]['lr'] = lr

        if self.opt.advent:
            advent_lr = max(self.opt.advent_lr * ((1 - float(global_iteration) / total_iterations) ** (self.opt.power)), min_lr)
            self.opt_d.param_groups[0]['lr'] = advent_lr
        return lr

    def next_tgt_batch(self):
        try:
            return next(self.loader_iter_tgt)
        except StopIteration:
            self.loader_iter_tgt = iter(self.target_dataloader)
            return next(self.loader_iter_tgt)

    def preprocess_data(self, data):
        if self.opt.sample_fixed_crop is not None:
            # new_img = []
            # new_sem = []

            h = int(self.opt.dim)
            w = int(self.opt.dim * self.opt.aspect_ratio)
            h_crop = self.opt.sample_fixed_crop[0]
            w_crop = self.opt.sample_fixed_crop[1]
            max_zoom = 1. # self.opt.max_zoom
            zoom = self.opt.min_zoom + random.random() * (max_zoom - self.opt.min_zoom)
            h_scaled = int(h * zoom)
            w_scaled = int(w * zoom)
            scale = (h_scaled, w_scaled)
            assert h_scaled - h_crop >= 0
            assert w_scaled - w_crop >= 0
            top_crop = int(random.random() * (h_scaled - h_crop))
            left_crop = int(random.random() * (w_scaled - w_crop))
            data["img"] = torch.nn.functional.interpolate(data["img"], size=scale, mode="bilinear")
            data["img"] = data["img"][:, :, top_crop:top_crop + h_crop, left_crop:left_crop + w_crop]
            data["sem_seg"] = torch.nn.functional.interpolate(data["sem_seg"], size=scale, mode="nearest")
            data["sem_seg"] = data["sem_seg"][:, :, top_crop:top_crop + h_crop, left_crop:left_crop + w_crop]
            return data
        if self.opt.sample_random_crop:
            h = int(self.opt.dim)
            w = int(self.opt.dim * self.opt.aspect_ratio)
            max_zoom = self.opt.max_zoom
            zoom = self.opt.min_zoom + random.random() * (max_zoom - self.opt.min_zoom)
            h_scaled = int(h * zoom)
            w_scaled = int(w * zoom)
            scale = (h_scaled, w_scaled)
            assert h_scaled - h >= 0
            assert w_scaled - w >= 0
            top_crop = int(random.random() * (h_scaled - h))
            left_crop = int(random.random() * (w_scaled - w))
            data["img"] = torch.nn.functional.interpolate(data["img"], size=scale, mode="bilinear")
            data["img"] = data["img"][:, :, top_crop:top_crop + h, left_crop:left_crop + w]
            data["sem_seg"] = torch.nn.functional.interpolate(data["sem_seg"], size=scale, mode="nearest")
            data["sem_seg"] = data["sem_seg"][:, :, top_crop:top_crop + h, left_crop:left_crop + w]
            return data
        else:
            return data

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            self.train_dataset = engine.create_dataset(self.all_opt, load_seg=True, load_img=True, is_synthetic=self.opt.synthetic_dataset, is_semi=self.opt.semi)
            self.train_dataloader, self.datasampler = engine.create_dataloader(self.train_dataset, self.opt.batch_size, self.opt.num_workers, is_train=True, is_synthetic=self.opt.synthetic_dataset)
            if self.opt.advent or self.opt.duo:
                self.target_dataset = engine.create_dataset(self.tgt_dataset_opt, load_seg=True, load_img=True)
                self.target_dataloader, self.target_datasampler = engine.create_dataloader(self.target_dataset, self.opt.batch_size, self.opt.num_workers, True)
                self.loader_iter_tgt = iter(self.target_dataloader)
            eval_opt = self.opt if self.opt.eval_dataset == "base" else self.tgt_dataset_opt
            self.valid_dataset = engine.create_dataset(eval_opt, load_seg=True, load_img=True, phase="valid")
            eval_batch_size = self.opt.force_eval_batch_size if self.opt.force_eval_batch_size is not None else self.opt.batch_size
            if not self.opt.no_eval:
                self.valid_dataloader, _ = engine.create_dataloader(self.valid_dataset, eval_batch_size, self.opt.num_workers, False)
            is_main = self.opt.local_rank == 0
            logger = Logger(self.opt) if is_main else None
            if self.opt.plus:
                self.segmentor_on_one_gpu = SegmentorPlus(self.opt, is_train=True, is_main=is_main, logger=logger, distributed=self.engine.distributed)
            else:
                self.segmentor_on_one_gpu = Segmentor(self.opt, is_train=True, is_main=is_main, logger=logger, distributed=self.engine.distributed)
            self.opt_s = self.segmentor_on_one_gpu.opt_s
            self.opt_d = self.segmentor_on_one_gpu.opt_d
            if self.opt.use_amp:
                optimizer = [self.opt_s, self.opt_d] if self.opt.advent else self.opt_s
                self.segmentor_on_one_gpu, optimizer = amp.initialize(self.segmentor_on_one_gpu, optimizer,
                                                                      opt_level=self.opt.amp_level)
                self.segmentor_on_one_gpu.apply(lambda x: cast_running_stats(x, self.engine.distributed))
            self.segmentor = engine.data_parallel(self.segmentor_on_one_gpu)
            self.segmentor.train()

            if self.opt.cont_train:
                start_epoch = self.opt.which_iter
            else:
                start_epoch = 0
            end_epoch = self.opt.niter

            self.dataset_size = len(self.train_dataset) * self.opt.batch_size if self.opt.synthetic_dataset else len(self.train_dataset)
            global_iteration = start_epoch * int(self.dataset_size / self.opt.batch_size)

            for epoch in range(start_epoch, end_epoch):
                if self.engine.distributed:
                    self.datasampler.set_epoch(epoch)
                    if self.opt.advent:
                        self.target_datasampler.set_epoch(epoch)
                for i, data_i in enumerate(self.train_dataloader):
                    global_iteration += 1
                    log = global_iteration % self.opt.log_freq == 0 and is_main
                    data_i = self.preprocess_data(data_i)
                    self.step_model(data_i, global_iteration, log=log)

                    lr = self.update_learning_rate(global_iteration)
                    if log:
                        self.segmentor_on_one_gpu.logger.log_scalar("segmentor/learning_rate", lr, global_iteration)
                        print(f"[Ep{epoch}/{end_epoch}] Iteration {i + 1:05d}/{int(self.dataset_size / self.opt.batch_size):05d}")

                # update sampling weights of synthetic dataset
                if self.opt.synthetic_dataset and not self.opt.semi:
                    self.train_dataset.update_sampler(logger=logger, log=is_main, global_iteration=global_iteration)

                if self.opt.save_freq > 0 and epoch % self.opt.save_freq == 0 and is_main:
                    self.segmentor_on_one_gpu.save_model(epoch, latest=False)
                if self.opt.save_latest_freq > 0 and epoch % self.opt.save_latest_freq == 0 and is_main:
                    self.segmentor_on_one_gpu.save_model(epoch, latest=True)

                if epoch % self.opt.eval_freq == 0 and not self.opt.no_eval:
                    self.segmentor.eval()
                    with torch.no_grad():
                        confusion_matrix = torch.zeros((self.opt.num_semantics, self.opt.num_semantics)).cuda()
                        for i, data_i in tqdm(enumerate(self.valid_dataloader), desc='eval', total=len(self.valid_dataloader)):
                            confusion_matrix += self.eval_model(data_i, global_iteration, log=i == 0)
                        confusion_matrix = self.engine.all_reduce_tensor(confusion_matrix, norm=False)

                        pos = confusion_matrix.sum(dim=1)
                        res = confusion_matrix.sum(dim=0)
                        tp = torch.diag(confusion_matrix)
                        iou = (tp / torch.max(torch.Tensor([1.0]).to(pos.get_device()), pos + res - tp))
                        mean_iou = iou.mean()

                        pos_eval = pos[self.opt.eval_idx]
                        res_eval = confusion_matrix[self.opt.eval_idx].sum(dim=0)[self.opt.eval_idx]
                        tp_eval = tp[self.opt.eval_idx]
                        iou_eval = (tp_eval / torch.max(torch.Tensor([1.0]).to(pos.get_device()), pos_eval + res_eval - tp_eval))
                        mean_iou_eval = iou_eval.mean()

                        if is_main:
                            self.segmentor_on_one_gpu.logger.log_scalar("segmentor/val/mean_iou", mean_iou, global_iteration)
                            self.segmentor_on_one_gpu.logger.log_scalar("segmentor/val/mean_iou_eval", mean_iou_eval, global_iteration)
                            for i in range(len(iou)):
                                self.segmentor_on_one_gpu.logger.log_scalar(f"segmentor/val/iou/{self.opt.semantic_labels[i].replace(' ', '_')}", iou[i], global_iteration)
                            self.segmentor_on_one_gpu.logger.log_confusion_matrix("segmentor/val/confusion_matrix", confusion_matrix.cpu(), global_iteration)
                    self.segmentor.train()

            print('Training was successfully finished.')


def cast_running_stats(m, distributed):
    ABN = InPlaceABNSync if distributed else InPlaceABN
    if isinstance(m, ABN):
        m.running_mean = m.running_mean.float()
        m.running_var = m.running_var.float()

if __name__ == "__main__":
    opt = Options().parse(load_segmentor=True, load_seg_generator=True, load_img_generator=True, load_extra_dataset=True, save=True)
    SegmentorTrainer(opt).run()
