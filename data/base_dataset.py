import os
import random
import time
import PIL
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


from data.utils import get_gaussian, get_coord, get_semantic_features, get_gaussian_filter, get_soft_sem_seg
from data.utils import get_instance_center_offset, get_instance_density, get_instance_cond, get_instance_edge
from tools.cond_estimator import CondEstimator
from tools.nearest_cond_indexor import NearestCondIndexor

class BaseDataset(data.Dataset):
    def __init__(self, load_seg, load_img, opt, phase='train'):
        super(BaseDataset, self).__init__()
        self.load_seg = load_seg
        self.load_img = load_img
        self.opt = opt
        self.phase = phase

        seg_paths, ins_paths, img_paths = self.get_paths(opt, phase=phase)
        if not opt.not_sort or phase == 'valid':
            seg_paths = sorted(seg_paths)
            img_paths = sorted(img_paths)
            ins_paths = sorted(ins_paths) if ins_paths is not None else ins_paths

        seg_paths = seg_paths[:opt.max_dataset_size]
        img_paths = img_paths[:opt.max_dataset_size]
        ins_paths = ins_paths[:opt.max_dataset_size] if ins_paths is not None else ins_paths

        if not opt.no_pairing_check:
            for path1, path2 in zip(seg_paths, img_paths):
                err = f"The label-image pair ({path1}, {path2}) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/urban_dataset.py to see what is going on, and use --no_pairing_check to bypass this."
                assert self.paths_match(path1, path2), err
            if ins_paths is not None:
                for path1, path2 in zip(ins_paths, img_paths):
                    err = f"The instance-image pair ({path1}, {path2}) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/urban_dataset.py to see what is going on, and use --no_pairing_check to bypass this."
                    assert self.paths_match(path1, path2), err

        self.seg_paths = seg_paths
        self.ins_paths = ins_paths
        self.img_paths = img_paths

        self.dataset_size = len(self.seg_paths)

        self.dim_ind = int(np.log2(opt.dim)) - 2
        self.dim = [2 ** k for k in range(2, int(np.log2(opt.max_dim)) + 1)]

        if opt.load_panoptic:
            self.sigma = [max(d * opt.max_sigma / opt.max_dim, opt.min_sigma) for d in self.dim]
            self.g = [get_gaussian(sigma) for sigma in self.sigma]
            self.x_coord, self.y_coord = get_coord(opt.max_dim, opt.aspect_ratio)

        if opt.soft_sem_seg:
            sigma = [max(d * opt.max_sigma / opt.max_dim, opt.min_sigma) for d in self.dim]
            self.gaussian_filter = [get_gaussian_filter(s, opt.num_semantics, d) for s, d in zip(sigma, self.dim)]

        if "soft" in self.opt.instance_type:
            sigma = [max(d * opt.max_sigma / opt.max_dim, opt.min_sigma) for d in self.dim]
            self.edge_filter = [get_gaussian_filter(s, 1, d) for s, d in zip(sigma, self.dim)]

        if opt.estimated_cond:
            self.cond_estimator = CondEstimator(opt)
            assert self.cond_estimator.is_fitted(), "Cond estimator should be fitted before being used in dataset"
            if opt.nearest_cond_index and self.phase == "train":
                self.nearest_cond_indexor = NearestCondIndexor(opt)
                assert self.nearest_cond_indexor.is_fitted()

    def get_paths(self, opt):
        seg_paths = []
        ins_paths = []
        img_paths = []
        assert False, "A subclass of UrbanDataset must override self.get_paths(self, opt)"
        return seg_paths, ins_paths, img_paths

    def preprocess_seg(self, seg, ins):
        return seg

    def postprocess_seg(self, seg, ins):
        return seg

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def get_augmentation_parameters(self):
        v_flip = random.random() > 0.5 if self.phase == 'train' and not self.opt.no_v_flip else False
        h_flip = random.random() > 0.5 if self.phase == 'train' and not self.opt.no_h_flip else False
        h = int(self.opt.true_dim)
        w = int(self.opt.true_dim * self.opt.true_ratio)
        if self.opt.fixed_top_centered_zoom:
            h_crop = int(h / self.opt.fixed_top_centered_zoom)
            w_crop = int(h_crop * self.opt.aspect_ratio)
            top_crop = 0
            assert w >= w_crop
            left_crop = int((w - w_crop) / 2)
            scale = None
        elif self.opt.fixed_crop:
            h_crop = self.opt.fixed_crop[0] if self.phase == 'train' else h
            w_crop = self.opt.fixed_crop[1] if self.phase == 'train' else w
            zoom = self.opt.min_zoom + random.random() * (self.opt.max_zoom - self.opt.min_zoom) if self.phase == 'train' else 1.
            h_scaled = int(h * zoom)
            w_scaled = int(w * zoom)
            scale = (h_scaled, w_scaled)
            assert h_scaled - h_crop >= 0
            assert w_scaled - w_crop >= 0
            top_crop = int(random.random() * (h_scaled - h_crop)) if self.phase == 'train' else 0
            left_crop = int(random.random() * (w_scaled - w_crop)) if self.phase == 'train' else 0
        else:
            min_zoom = max(1., self.opt.aspect_ratio / self.opt.true_ratio)
            max_zoom = max(self.opt.max_zoom, min_zoom)
            zoom = min_zoom + random.random() * (max_zoom - min_zoom) if self.phase == 'train' else min_zoom
            h_crop = int(h / zoom)
            w_crop = int(h_crop * self.opt.aspect_ratio)
            assert h >= h_crop
            assert w >= w_crop
            top_crop = int(random.random() * (h - h_crop)) if self.phase == 'train' else 0
            left_crop = int(random.random() * (w - w_crop)) if self.phase == 'train' else 0
            scale = None
        return v_flip, h_flip, top_crop, left_crop, h_crop, w_crop, scale

    def __getitem__(self, index):
        time_tensor = torch.zeros(4)
        dim = self.dim[self.dim_ind]
        v_flip, h_flip, top_crop, left_crop, h_crop, w_crop, scale = self.get_augmentation_parameters()
        empty = torch.Tensor([])

        sem_seg, soft_sem_seg, sem_cond, flat_seg = empty, empty, empty, empty
        ins_density, ins_center, ins_offset, ins_edge, ins_cond = empty, empty, empty, empty, empty
        tgt_sem_cond, tgt_ins_cond = empty, empty
        seg, img = empty, empty

        t0 = time.time()

        if self.load_seg:
            if self.opt.load_minimal_info:
                t1 = time.time()
                seg_path = self.seg_paths[index].split(' ')[0]
                seg = PIL.Image.open(seg_path)
                seg = seg.transpose(method=PIL.Image.ROTATE_90) if self.opt.transpose else seg
                t2 = time.time()
                transform_seg = get_transform(dim, v_flip=v_flip, h_flip=h_flip, method=PIL.Image.NEAREST,
                                              normalize=False, top_crop=top_crop,
                                              left_crop=left_crop, h_crop=h_crop, w_crop=w_crop,
                                              resize=self.opt.resize_seg, scale=scale)
                t3 = time.time()
                seg = (transform_seg(seg) * 255).long()[0]
                flat_seg = seg
                t4 = time.time()
                # print(f"load seg {t2 - t1:02f}, get transform {t3 - t2:02f}, transform seg {t4 - t3:02f}")
            else:
                if self.opt.estimated_cond and self.phase == "train":
                    tgt_sem_cond, tgt_ins_cond = self.cond_estimator.sample()
                    tgt_sem_cond = tgt_sem_cond.squeeze()
                    tgt_ins_cond = tgt_ins_cond.squeeze()
                    if self.opt.nearest_cond_index and self.phase == "train":
                        index = self.nearest_cond_indexor.get_nearest_idx(tgt_sem_cond, tgt_ins_cond)

                t01 = time.time()

                seg_path = self.seg_paths[index].split(' ')[0]
                seg = PIL.Image.open(seg_path)
                seg = seg.transpose(method=PIL.Image.ROTATE_90) if self.opt.transpose else seg
                t1 = time.time()
                ins_path = self.ins_paths[index].split(' ')[0] if self.ins_paths is not None and self.opt.load_panoptic else None
                ins = PIL.Image.open(ins_path) if ins_path is not None else None
                ins = ins.transpose(method=PIL.Image.ROTATE_90) if self.opt.transpose and ins is not None else ins
                seg = self.preprocess_seg(seg, ins)
                t2 = time.time()
                dim_for_seg = self.opt.seg_dim if self.opt.force_seg_dim else dim
                transform_seg = get_transform(dim_for_seg, v_flip=v_flip, h_flip=h_flip, method=PIL.Image.NEAREST, normalize=False, top_crop=top_crop,
                                              left_crop=left_crop, h_crop=h_crop, w_crop=w_crop, resize=self.opt.resize_seg, scale=scale)
                t3 = time.time()
                seg = transform_seg(seg).float()[0]
                seg = self.postprocess_seg(seg, ins)
                sem_seg, sem_cond = get_semantic_features(seg, self.opt.num_semantics, self.opt.label_nc)

                t4 = time.time()
                if self.opt.soft_sem_seg:
                    sem_seg = get_soft_sem_seg(sem_seg, self.gaussian_filter[self.dim_ind], self.opt.soft_sem_prop)

                t5 = time.time()
                if self.opt.load_panoptic:
                    ins_cond = get_instance_cond(seg, self.opt.things_idx)
                    height, width = seg.shape[:2]
                    scaled_x_coord = self.x_coord[:height, :width] / height
                    scaled_y_coord = self.y_coord[:height, :width] / height
                    if "edge" in self.opt.instance_type:
                        gaussian_filter = self.edge_filter[self.dim_ind] if "soft_edge" in self.opt.instance_type else None
                        ins_edge = get_instance_edge(seg, gaussian_filter)
                    if "center_offset" in self.opt.instance_type:
                        ins_center, ins_offset = get_instance_center_offset(seg, self.g[self.dim_ind],
                                                                            self.sigma[self.dim_ind], scaled_x_coord,
                                                                            scaled_y_coord, height)
                    if "density" in self.opt.instance_type:
                        geometric = "geometric_density" in self.opt.instance_type
                        ins_density = get_instance_density(seg, self.opt.things_idx, scaled_x_coord, scaled_y_coord, height,
                                                           geometric, self.sigma[self.dim_ind])
                if self.opt.estimated_cond and self.phase == "train":
                    if not self.opt.has_tgt:
                        sem_cond, ins_cond = tgt_sem_cond, tgt_ins_cond
                else:
                    tgt_sem_cond, tgt_ins_cond = sem_cond, ins_cond
            t6 = time.time()

        if self.load_img:
            img_path = self.img_paths[index].split(' ')[0]
            img = PIL.Image.open(img_path)
            img = img.transpose(method=PIL.Image.ROTATE_90) if self.opt.transpose else img
            img = img.convert('RGB')
            transform_img = get_transform(dim, v_flip=v_flip, h_flip=h_flip, top_crop=top_crop, left_crop=left_crop,
                                          h_crop=h_crop, w_crop=w_crop, resize=self.opt.resize_img, scale=scale,
                                          imagenet=self.opt.imagenet_norm, colorjitter=self.opt.colorjitter)
            img = transform_img(img)

        if self.load_img and self.load_seg and not self.opt.no_pairing_check:
             assert self.paths_match(seg_path, img_path), f"The label_path {seg_path} and {img_path} don't match."
        t7 = time.time()
        # print(f"estimate cond: {t01 - t0:02f}, open seg: {t1 - t01:02f}, preprocess seg {t2 - t1:02f}, transform seg {t3 - t2:02f}, extract feat {t4 - t3:02f}, soften maps {t5 - t4:02f}, panoptic {t6 - t5:02f}, image {t7 - t6:02f}")
        # time_tensor[0] = t1 - t0
        # time_tensor[1] = t2 - t1
        # time_tensor[2] = t3 - t2
        # time_tensor[3] = t4 - t3

        input_dict = {'img': img,
                      'seg': seg,
                      'flat_seg': flat_seg,
                      'sem_seg': sem_seg,
                      'sem_cond': sem_cond,
                      'tgt_sem_cond': tgt_sem_cond,
                      'ins_center': ins_center,
                      'ins_offset': ins_offset,
                      'ins_edge': ins_edge,
                      'ins_density': ins_density,
                      'ins_cond': ins_cond,
                      'tgt_ins_cond': tgt_ins_cond,
                      'time': time_tensor}
        return input_dict

    def __len__(self):
        return self.dataset_size


def get_transform(dim, v_flip=False, h_flip=False, method=PIL.Image.BILINEAR, normalize=True, imagenet=False, top_crop=None, left_crop=None, h_crop=None, w_crop=None, resize=None, scale=None, colorjitter=False):
    transform_list = []
    if resize is not None:
        transform_list.append(transforms.Resize(resize, method))
    if scale is not None:
        transform_list.append(transforms.Resize(scale, method))
    if top_crop is not None:
        transform_list.append(transforms.Lambda(lambda img: transforms.functional.crop(img, top_crop, left_crop, h_crop, w_crop)))
    if scale is None:
        transform_list.append(transforms.Resize(dim, method))
    if v_flip:
        transform_list.append(transforms.Lambda(lambda img: img.transpose(PIL.Image.FLIP_LEFT_RIGHT)))
    if h_flip:
        transform_list.append(transforms.Lambda(lambda img: img.transpose(PIL.Image.FLIP_TOP_BOTTOM)))
    if colorjitter:
        transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5))
    transform_list.append(lambda x: transforms.ToTensor()(np.array(x)))
    if normalize:
        if imagenet:
            transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        else:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)
