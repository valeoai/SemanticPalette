import ot
import numpy as np
import torch
from copy import  deepcopy
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from math import ceil
from apex import amp

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urbangan_dir = os.path.dirname(current_dir)
sys.path.insert(0, urbangan_dir)

from tools.options import Options
from tools.engine import Engine
from tools.logger import Logger
from tools.fid.inception import InceptionV3
from tools.fid.fid_score import calculate_fid_given_acts, get_activations
from tools.fsd.fsd_score import calculate_fsd_given_props
from data import create_dataset, create_dataloader
from tools.utils import get_confusion_matrix
from models.segmentor.models.segmentor import Segmentor
from inplace_abn import InPlaceABN, InPlaceABNSync

class Tester:
    def __init__(self, opt):
        self.opt = opt["segmentor"]
        self.tgt_dataset_opt = opt["extra_dataset"]
        self.all_opt = opt

    def bhattacharyya(self, a, b, eps=0.000001):
        # https://en.wikipedia.org/wiki/Bhattacharyya_distance
        return -np.log(np.sum(np.sqrt(a * b)) + eps)

    def compute_emd(self, tgt_cond, cond):
        #M = ot.dist(tgt_cond, cond, metric="euclidean")
        M = cdist(tgt_cond, cond, self.bhattacharyya)
        tgt_n, n = tgt_cond.shape[0], cond.shape[0]
        tgt_dis, dis = np.ones((tgt_n,)) / tgt_n, np.ones((n,)) / n  # uniform distribution on samples
        emd = ot.emd2(tgt_dis, dis, M, numItermax=1000000)
        return emd

    def compute_acts(self, img, size):
        with torch.no_grad():
            acts = get_activations(img, self.inception_model, size, cuda=True)
        return acts

    def compute_fid(self, tgt_acts, acts):
        fid_eval = calculate_fid_given_acts(tgt_acts, acts)
        return fid_eval

    def compute_fsd(self, tgt_props, props):
        fsd_eval = calculate_fsd_given_props(tgt_props, props)
        return fsd_eval

    def compute_confusion_matrix(self, data, model):
        with torch.no_grad():
            if self.opt.slide_eval:
                pred_seg = self.slide_pred(model, data, window_size=self.opt.fixed_crop)
                if self.opt.multi_scale_eval:
                    if self.opt.aspect_ratio > 1:
                        size = [self.opt.fixed_crop[0], int(self.opt.fixed_crop[0] * self.opt.aspect_ratio)]
                    else:
                        size = [int(self.opt.fixed_crop[1] / self.opt.aspect_ratio), self.opt.fixed_crop[1]]
                    small_data = {"img": torch.nn.functional.interpolate(data["img"], size=size, mode="bilinear")}
                    pred_small_seg = self.slide_pred(model, small_data, window_size=self.opt.fixed_crop)
                    resized_seg = torch.nn.functional.interpolate(pred_small_seg["sem_seg"], size=data["img"].shape[-2:], mode="bilinear")
                    pred_seg["sem_seg"] = (pred_seg["sem_seg"] + resized_seg) / 2
            else:
                pred_seg = model(data, mode='inference', hard=False)
        pred_sem_seg = pred_seg["sem_seg"]
        sem_index_pred = pred_sem_seg.max(dim=1, keepdim=True)[1]
        real_sem_seg = data["sem_seg"].cuda()
        sem_index_real = real_sem_seg.max(dim=1, keepdim=True)[1]
        confusion_matrix = get_confusion_matrix(sem_index_real, sem_index_pred, self.opt.num_semantics)
        return confusion_matrix.cpu()

    def slide_pred(self, model, data, window_size):
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
                pred = model(slide_data, mode='inference', hard=False)
                pred_sem_seg[:, :, y1:y2, x1:x2] = pred["sem_seg"]
                pred_sem_count[y1:y2, x1:x2] += 1
        pred_sem_seg /= pred_sem_count
        return {"sem_seg": pred_sem_seg}

    def compute_iou(self, confusion_matrix):
        pos = confusion_matrix.sum(dim=1)
        res = confusion_matrix.sum(dim=0)
        tp = torch.diag(confusion_matrix)
        iou = (tp / torch.max(torch.Tensor([1.0]), pos + res - tp))
        mean_iou = iou.mean()
        pos_eval = pos[self.opt.eval_idx]
        res_eval = confusion_matrix[self.opt.eval_idx].sum(dim=0)[self.opt.eval_idx]
        tp_eval = tp[self.opt.eval_idx]
        iou_eval = (tp_eval / torch.max(torch.Tensor([1.0]), pos_eval + res_eval - tp_eval))
        mean_iou_eval = iou_eval.mean()
        return mean_iou, mean_iou_eval, iou, iou_eval

    def kl(self, p, q, eps=0.000001):
        q = q + eps
        q = 1.0 * q / np.sum(q, axis=1, keepdims=True)
        return np.mean(np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=1))

    def js(self, p, q):
        m = 0.5 * p + 0.5 * q
        return 0.5 * self.kl(p, m) + 0.5 * self.kl(q, m)

    def compute_prop_error(self, gt_cond, cond, thresh=0.01):
        if len(gt_cond > 0):
            kl = self.kl(gt_cond, cond)
            js = self.js(gt_cond, cond)
            ae = np.abs(gt_cond - cond)
            class_mae = np.mean(ae, axis=0)
            thresh_mask = gt_cond > thresh
            num_mask = np.sum(thresh_mask.astype(float), axis=0)
            num_mask[num_mask == 0] = 1
            class_thresh_mae = np.sum(ae * thresh_mask.astype(float), axis=0) / num_mask
            mae = np.mean(ae)
            tmae = np.mean(class_thresh_mae)
        else:
            e = -1
            l = [e for j in range(len(self.opt.semantic_labels))]
            kl, js, mae, tmae = e, e, e, e
            class_mae, class_thresh_mae = l, l
        return kl, js, mae, tmae, class_mae, class_thresh_mae

    def save_to_csv(self, res, test_iou, test_iou_eval, train_iou, train_iou_eval, per_class_cond_dis):
        log_dir = os.path.join(self.opt.save_path, "logs")
        name = self.opt.dataset
        if self.opt.load_extra:
            name += "_extra"
        csv_file = os.path.join(log_dir, f"generator_statistics_{name}.csv")
        excel_file = os.path.join(log_dir, f"generator_statistics_{name}.xlsx")
        eval_labels = [self.opt.semantic_labels[i] for i in self.opt.eval_idx]
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, sep=',')
        else:
            columns = ["method", "log_likelihood", "emd", "fid", "test_miou", "test_miou_eval", "train_miou",
                       "train_miou_eval", "kl", "js", "mae", "tmae", "d_log_likelihood", "d_emd", "d_fid", "d_test_miou",
                       "d_test_miou_eval", "d_train_miou", "d_train_miou_eval", "d_kl", "d_js", "d_mae", "d_tmae",
                       "fsd", "d_fsd", "fsd_count", "d_fsd_count"]
            columns += [f"test_iou_{lbl}" for lbl in self.opt.semantic_labels]
            columns += [f"test_iou_eval_{lbl}" for lbl in eval_labels]
            columns += [f"train_iou_{lbl}" for lbl in self.opt.semantic_labels]
            columns += [f"train_iou_eval_{lbl}" for lbl in eval_labels]
            columns += [f"fsd_{lbl}" for lbl in self.opt.semantic_labels]
            df = pd.DataFrame(columns=columns)
        new_row = {}
        for key in res:
            new_row[key] = np.mean(res[key])
            new_row["d_" + key] = np.std(res[key])
        for i, lbl in enumerate(self.opt.semantic_labels):
            new_row[f"train_iou_{lbl}"] = train_iou[i]
            new_row[f"test_iou_{lbl}"] = test_iou[i]
            new_row[f"fsd_{lbl}"] = per_class_cond_dis[i]
        for i, lbl in enumerate(eval_labels):
            new_row[f"train_iou_eval_{lbl}"] = train_iou_eval[i]
            new_row[f"test_iou_eval_{lbl}"] = test_iou_eval[i]
        new_row["method"] = self.opt.name
        df = df.append(new_row, ignore_index=True)
        df.to_csv(csv_file, sep=",", index=False)
        df.to_excel(excel_file)

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            self.dataset = engine.create_dataset(self.all_opt, load_seg=True, load_img=True, is_synthetic=self.opt.synthetic_dataset, is_semi=self.opt.semi)
            self.dataloader, self.datasampler = engine.create_dataloader(self.dataset, self.opt.batch_size, self.opt.num_workers, is_train=True, is_synthetic=self.opt.synthetic_dataset)
            self.tgt_dataset = create_dataset(self.tgt_dataset_opt, load_seg=True, load_img=True, phase='valid')
            self.tgt_dataloader = create_dataloader(self.tgt_dataset, self.opt.batch_size, self.opt.num_workers, is_train=False)
            self.epoch = 0
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception_model = InceptionV3([block_idx])
            self.inception_model.cuda()
            is_main = self.opt.local_rank == 0
            logger = Logger(self.opt) if is_main else None
            self.segmentor_on_one_gpu = Segmentor(self.opt, is_train=False, is_main=is_main, logger=logger, distributed=self.engine.distributed)
            if self.opt.use_amp:
                self.segmentor_on_one_gpu = amp.initialize(self.segmentor_on_one_gpu, opt_level="O1")
                self.segmentor_on_one_gpu.apply(lambda x: cast_running_stats(x, self.engine.distributed))
            self.segmentor = engine.data_parallel(self.segmentor_on_one_gpu)
            self.segmentor.eval()
            if self.opt.load_path_2 is not None:
                opt_2 = deepcopy(self.opt)
                opt_2.which_iter = self.opt.which_iter_2
                opt_2.load_path = self.opt.load_path_2
                self.segmentor_on_one_gpu_2 = Segmentor(opt_2, is_train=False, is_main=is_main, logger=logger, distributed=self.engine.distributed)
                if self.opt.use_amp:
                    self.segmentor_on_one_gpu_2 = amp.initialize(self.segmentor_on_one_gpu_2, opt_level="O1")
                    self.segmentor_on_one_gpu_2.apply(lambda x: cast_running_stats(x, self.engine.distributed))
                self.segmentor_2 = engine.data_parallel(self.segmentor_on_one_gpu_2)
                self.segmentor_2.eval()
            else:
                self.segmentor_2 = None

            cond_codes = []
            tgt_cond_codes = []
            gt_cond_codes = []
            all_acts = []
            all_tgt_acts = []
            train_confusion_matrix = torch.zeros((self.opt.num_semantics, self.opt.num_semantics))
            test_confusion_matrix = torch.zeros((self.opt.num_semantics, self.opt.num_semantics))

            for k, data in enumerate(tqdm(self.tgt_dataloader, desc="Retrieving tgt prop")):
                sem = data["sem_seg"]
                index = sem.max(1, keepdim=True)[1]
                d_sem = torch.zeros_like(sem).scatter_(1, index, 1.0)
                d_sem_cond = torch.mean(d_sem, dim=(2, 3))
                tgt_cond_codes.append(d_sem_cond.cpu().numpy())
                img = data["img"]
                size = img.shape[0]
                all_tgt_acts.append(self.compute_acts(img.cuda(), size))
                # if self.segmentor_2 is not None:
                #     train_confusion_matrix += self.compute_confusion_matrix(data, self.segmentor_2)
                if k * self.opt.batch_size > 5000:
                    break

            train_mean_iou, train_mean_iou_eval, train_iou, train_iou_eval = self.compute_iou(train_confusion_matrix)
            train_confusion_matrix_eval = train_confusion_matrix[self.opt.eval_idx][:, self.opt.eval_idx]
            print("train_mean_iou:", train_mean_iou, ", train_mean_iou_eval:", train_mean_iou_eval)
            tgt_cond = np.concatenate(tgt_cond_codes)
            tgt_acts = np.concatenate(all_tgt_acts)

            fit_score = []
            emd = []
            fid = []
            test_mean_iou = []
            test_mean_iou_eval = []
            kl = []
            js = []
            mae = []
            tmae = []
            fsd = []
            fsd_count = []
            # pxl_count = self.opt.dim * self.opt.dim * self.opt.aspect_ratio

            for i in range(self.opt.niter):
                if self.engine.distributed:
                    self.datasampler.set_epoch(i)
                for k, data in enumerate(tqdm(self.dataloader, desc=f"[{i+1}/{self.opt.niter}] Retrieving generated prop")):
                    sem = data["sem_seg"]
                    index = sem.max(1, keepdim=True)[1]
                    d_sem = torch.zeros_like(sem).scatter_(1, index, 1.0)
                    d_sem_cond = torch.mean(d_sem, dim=(2, 3))
                    cond_codes.append(d_sem_cond.cpu().numpy())
                    sem_cond = deepcopy(data["sem_cond"])
                    gt_cond_codes.append(sem_cond.cpu().numpy())
                    img = data["img"]
                    size = img.shape[0]
                    all_acts.append(self.compute_acts(img.cuda(), size))
                    # test_confusion_matrix += self.compute_confusion_matrix(data, self.segmentor)
                    del sem
                    del img
                    del sem_cond
                    if k * self.opt.batch_size > 5000:
                        break
                if is_main:
                    cond = np.concatenate(cond_codes)
                    gt_cond = np.concatenate(gt_cond_codes)
                    fit_score.append(self.tgt_dataset.cond_estimator.gmm.score(cond))
                    emd.append(self.compute_emd(tgt_cond, cond))
                    fsd.append(self.compute_fsd(tgt_cond, cond))
                    per_class_cond_dis = np.mean(100 * tgt_cond, axis=0) - np.mean(100 * cond, axis=0)
                    fsd_count.append(self.compute_fsd(tgt_cond * 100, cond * 100))
                    prop_error = self.compute_prop_error(gt_cond, cond)
                    kl.append(prop_error[0])
                    js.append(prop_error[1])
                    mae.append(prop_error[2])
                    tmae.append(prop_error[3])
                    class_mae = prop_error[4]
                    class_thresh_mae = prop_error[5]
                    acts = np.concatenate(all_acts)
                    fid.append(self.compute_fid(tgt_acts, acts))
                    iou = self.compute_iou(test_confusion_matrix)
                    test_mean_iou.append(iou[0])
                    test_mean_iou_eval.append(iou[1])
                    test_iou = iou[2]
                    test_iou_eval = iou[3]
                    test_confusion_matrix_eval = test_confusion_matrix[self.opt.eval_idx][:, self.opt.eval_idx]
                    logger.log_scalar("generator_statistics/avg_log_likelihood", np.mean(fit_score), i)
                    logger.log_scalar("generator_statistics/emd", np.mean(emd), i)
                    logger.log_scalar("generator_statistics/fsd", np.mean(fsd), i)
                    logger.log_scalar("generator_statistics/fid", np.mean(fid), i)
                    logger.log_scalar("generator_statistics/gan_test/mean_iou", np.mean(test_mean_iou), i)
                    logger.log_scalar("generator_statistics/gan_test/mean_iou_eval", np.mean(test_mean_iou_eval), i)
                    logger.log_scalar("generator_statistics/gan_train/mean_iou", train_mean_iou, i)
                    logger.log_scalar("generator_statistics/gan_train/mean_iou_eval", train_mean_iou_eval, i)
                    logger.log_scalar("generator_statistics/kl", np.mean(kl), i)
                    logger.log_scalar("generator_statistics/js", np.mean(js), i)
                    logger.log_scalar("generator_statistics/mae", np.mean(mae), i)
                    logger.log_scalar("generator_statistics/tmae", np.mean(tmae), i)
                    for j in range(len(self.opt.semantic_labels)):
                        logger.log_scalar(f"generator_statistics/gan_test/iou/{self.opt.semantic_labels[j].replace(' ', '_')}", test_iou[j], i)
                        logger.log_scalar(f"generator_statistics/gan_train/iou/{self.opt.semantic_labels[j].replace(' ', '_')}", train_iou[j], i)
                        logger.log_scalar(f"generator_statistics/mae/{self.opt.semantic_labels[j].replace(' ', '_')}", class_mae[j], i)
                        logger.log_scalar(f"generator_statistics/thresh_mae/{self.opt.semantic_labels[j].replace(' ', '_')}", class_thresh_mae[j], i)
                    logger.log_confusion_matrix("generator_statistics/gan_test/confusion_matrix", test_confusion_matrix, i)
                    logger.log_confusion_matrix("generator_statistics/gan_test/confusion_matrix_eval", test_confusion_matrix_eval, i, eval_only=True)
                    logger.log_confusion_matrix("generator_statistics/gan_train/confusion_matrix", train_confusion_matrix, i)
                    logger.log_confusion_matrix("generator_statistics/gan_train/confusion_matrix_eval", train_confusion_matrix_eval, i, eval_only=True)
                    print(f"[{i+1}/{self.opt.niter}] average log likelihood: {np.mean(fit_score)}, emd: {np.mean(emd)}, fsd: {np.mean(fsd)}, fsd count: {np.mean(fsd_count)}, fid: {np.mean(fid)}, [train] miou: {train_mean_iou}, [train] miou (eval): {train_mean_iou_eval}, [test] miou: {np.mean(test_mean_iou)}, [test] miou (eval): {np.mean(test_mean_iou_eval)}, kl: {np.mean(kl)}, js: {np.mean(js)}, mae: {np.mean(mae)}")
                    if i == self.opt.niter - 1:
                        print("Saving to csv")
                        self.save_to_csv({"log_likelihood": fit_score, "emd":emd, "fid":fid, "test_miou":test_mean_iou,
                                          "test_miou_eval":test_mean_iou_eval, "train_miou":[train_mean_iou],
                                          "train_miou_eval":[train_mean_iou_eval], "kl":kl, "js":js, "mae":mae, "tmae":tmae, "fsd":fsd, "fsd_count":fsd_count},
                                         test_iou.numpy(), test_iou_eval.numpy(), train_iou.numpy(), train_iou_eval.numpy(), per_class_cond_dis)
            print('Statistics successfully computed.')


def cast_running_stats(m, distributed):
    ABN = InPlaceABNSync if distributed else InPlaceABN
    if isinstance(m, ABN):
        m.running_mean = m.running_mean.float()
        m.running_var = m.running_var.float()


if __name__ == "__main__":
    opt = Options().parse(load_segmentor=True, load_seg_generator=True, load_img_generator=True, load_extra_dataset=True, save=True)
    Tester(opt).run()
