from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np
import random
import itertools

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from data.utils import get_coord
from tools.utils import create_colormap, color_transfer, color_spread

class Logger():
    def __init__(self, opt):
        self.writer = SummaryWriter(opt.log_path)
        self.colormap = create_colormap(opt)
        self.num_semantics = opt.num_semantics
        self.semantic_labels = opt.semantic_labels
        self.num_things = opt.num_things
        self.things_idx = opt.things_idx
        self.x_coord, self.y_coord = get_coord(opt.max_dim, opt.aspect_ratio)
        self.log_path = opt.log_path
        self.eval_idx = opt.eval_idx

    def is_empty(self, tensors):
        for tensor in tensors:
            if 0 in tensor.size():
                return True
        return False
    
    def log_img(self, name, tensor, nrow, global_iteration, normalize=False, range=None, pad_value=0):
        if self.is_empty([tensor]):
            return
        with torch.no_grad():
            grid = make_grid(tensor, nrow=nrow, normalize=normalize, range=range, pad_value=pad_value)
            self.writer.add_image(name, grid, global_iteration)
    
    def log_scalar(self, name, scalar, global_iteration):
        if scalar is not None:
            if type(scalar) == list:
                for i, x in enumerate(scalar):
                    self.log_scalar(f"{name}_{i}", x, global_iteration)
            else:
                self.writer.add_scalar(name, scalar, global_iteration)
    
    def log_spread(self, name, spread, nrow, global_iteration, max_spread=5):
        if self.is_empty([spread]):
            return
        with torch.no_grad():
            im_spread_rgb = color_spread(spread, max_spread)
            self.log_img(name + f"-max{max_spread}", im_spread_rgb, nrow, global_iteration)
    
    def log_semantic_mask(self, name, semantic_mask, sem_cond, mask_num, nrow, global_iteration, thresh=0.001):
        if self.is_empty([semantic_mask, sem_cond]):
            return
        with torch.no_grad():
            semantic_mask = semantic_mask[0]
            sem_cond = sem_cond[0]
            possible_idx = list((sem_cond != 0).nonzero().flatten().numpy())
            num = min(len(possible_idx), mask_num)
            idx = sorted(random.sample(possible_idx, num))
            semantic_mask = semantic_mask[idx]
            semantic_mask = semantic_mask.view(num, 1, *semantic_mask.size()[1:])
            masks_max = torch.max(semantic_mask.view(num, 1, -1), dim=2)[0].view(num, 1, 1, 1)
            semantic_mask /= masks_max
            colors = torch.FloatTensor(num, 1, *semantic_mask.size()[2:]).zero_()
            for i in range(num):
                colors[i] = float(idx[i])
            colors = color_transfer(colors, self.colormap).cpu() / 2 + 0.5
            colors = colors * semantic_mask
            cell = max(semantic_mask.size(2) // 60, 1)
            background = torch.zeros(semantic_mask.size(2), semantic_mask.size(3))
            for i in range(cell):
                background[i::2 * cell] += 1
                background[:, i::2 * cell] += 1
            background = (background % 2) * 0.5 + 0.5
            background = background.repeat(3, 1, 1)
            for i in range(semantic_mask.size(0)):
                bg_mask = (semantic_mask[i] < thresh).repeat(3, 1, 1)
                colors[i][bg_mask] = background[bg_mask]
            self.log_img(name + f"-thresh{thresh}", colors, nrow, global_iteration)
    
    def log_semantic_seg(self, name, seg_mc, nrow, global_iteration):
        if self.is_empty([seg_mc]):
            return
        with torch.no_grad():
            seg = seg_mc if seg_mc.size(1) == 1 else seg_mc.max(1, keepdim=True)[1]
            seg[seg > self.num_semantics - 1] = -1
            seg = color_transfer(seg, self.colormap)
            self.log_img(name, seg, nrow, global_iteration, normalize=True, range=(-1, 1))

    def log_cond_distrib(self, name, real_cond, fake_cond, nrow, ncol, global_iteration, width=0.5):
        if self.is_empty([real_cond, fake_cond]):
            return
        with torch.no_grad():
            num = fake_cond.size(0)
            fake_cond = fake_cond.detach().numpy()
            real_cond = real_cond.detach().numpy()
            x = np.array(range(fake_cond.shape[1]))
            fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(7, 6))
            for i, ax in enumerate(axes.flat):
                if i < num:
                    ax.bar(x - width / 2, real_cond[i], width, label='real')
                    ax.bar(x + width / 2, fake_cond[i], width, label='fake')
                if i == num - 1:
                    ax.legend()
            fig.tight_layout()
            self.writer.add_figure(name, fig, global_iteration, close=True)

    def log_ins_center(self, name, ins_center, nrow, global_iteration):
        if self.is_empty([ins_center]):
            return
        with torch.no_grad():
            ins_center = torch.clamp(ins_center, max=1)
            self.log_img(name, ins_center, nrow, global_iteration, pad_value=1)

    def log_instance(self, name, seg_mc, center_mask, ins_offset, nrow, global_iteration):
        if self.is_empty([seg_mc, center_mask, ins_offset]):
            return
        with torch.no_grad():
            height, width = seg_mc.shape[2:]
            seg_mc_one_hot = torch.zeros_like(seg_mc).scatter_(1, seg_mc.max(dim=1, keepdim=True)[1], 1.0)
            instance_colors = torch.zeros((seg_mc.shape[0], *seg_mc.shape[2:], 3))
            scaled_x_coord = self.x_coord[:height, :width] / height
            scaled_y_coord = self.y_coord[:height, :width] / height
            shifted_x_coord = ins_offset[:, 0] + scaled_x_coord.view(1, height, width)
            shifted_y_coord = ins_offset[:, 1] + scaled_y_coord.view(1, height, width)
            for b in range(seg_mc.shape[0]):
                for i, k in enumerate(self.things_idx):
                    mask = (seg_mc_one_hot[b, k] > 0)
                    mask_shape = torch.sum(mask.long())
                    closest_dis = 10000 * torch.ones(mask_shape)
                    new_colors = torch.ones((mask_shape, 3))
                    for j, (center_y, center_x) in enumerate(center_mask[b, 0].nonzero()):
                        dis_x = shifted_x_coord[b][mask] - 1. * center_x / height
                        dis_y = shifted_y_coord[b][mask] - 1. * center_y / height
                        squared_dis_to_center = dis_x ** 2 + dis_y ** 2
                        new_center_mask = (squared_dis_to_center < closest_dis)
                        closest_dis[new_center_mask] = squared_dis_to_center[new_center_mask]
                        new_colors[new_center_mask] = 0.2 + 0.6 * torch.rand(3)
                    instance_colors[b][mask] = new_colors
            instance_colors = instance_colors.permute(0, 3, 1, 2)
            self.log_img(name, instance_colors, nrow, global_iteration, pad_value=1)

    def log_ins_offset(self, name, seg_mc, ins_offset, nrow, global_iteration):
        if self.is_empty([seg_mc, ins_offset]):
            return
        with torch.no_grad():
            index = seg_mc.max(dim=1, keepdim=True)[1]
            seg_mc = torch.zeros_like(seg_mc).scatter_(1, index, 1.0)
            bg = (seg_mc[:,self.things_idx].sum(dim=1) == 0)
            angle = (1 + torch.atan2(ins_offset[:, 1], ins_offset[:, 0]) / np.pi) / 2
            sat_norm = torch.min(10 * (torch.sqrt(ins_offset[:, 0] ** 2 + ins_offset[:, 1] ** 2)), torch.tensor([1.]))
            cmp = cm.get_cmap('hsv', 128)
            offset_rgba = cmp(angle.numpy())
            offset_rgb = torch.tensor(offset_rgba[:, :, :, :3]).float()
            offset_rgb = sat_norm.unsqueeze(-1) * offset_rgb + (1 - sat_norm).unsqueeze(-1) * torch.ones_like(offset_rgb)
            offset_rgb[bg] = torch.tensor([0., 0., 0.])
            offset_rgb = offset_rgb.permute(0, 3, 1, 2)
            self.log_img(name, offset_rgb, nrow, global_iteration, pad_value=1)

    def log_ins_density(self, name, ins_density, nrow, global_iteration):
        if self.is_empty([ins_density]):
            return
        with torch.no_grad():
            colored_mask = torch.zeros((ins_density.shape[0], *ins_density.shape[2:]))
            has_density = torch.sum(ins_density, dim=1) > 0
            max_density, idx = torch.max(ins_density, dim=1)
            max_max_density = torch.max(max_density.view(ins_density.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1)
            max_max_density[max_max_density == 0] = 1
            max_density /= max_max_density
            colored_mask[has_density] = torch.Tensor(self.things_idx)[idx[has_density]]
            colored_mask = color_transfer(colored_mask.unsqueeze(1), self.colormap)
            colored_mask = (colored_mask + 1) * max_density.unsqueeze(1) - 1
            self.log_img(name, colored_mask, nrow, global_iteration, normalize=True, range=(-1, 1))

    def log_confusion_matrix(self, name, confusion_matrix, global_iteration, save=False, eval_only=False):
        if self.is_empty([confusion_matrix]):
            return
        with torch.no_grad():
            num_sem_classes = len(self.eval_idx) if eval_only else self.num_semantics
            sem_labels = [self.semantic_labels[i] for i in self.eval_idx] if eval_only else self.semantic_labels
            cm = confusion_matrix.numpy()
            tot_gt = np.sum(cm, axis=1)
            tot_gt[tot_gt == 0] = 1
            n_cm = (cm.T / tot_gt).T
            fig = plt.figure(figsize=(int(0.6 * num_sem_classes), int(0.6 * num_sem_classes)))
            ax = fig.add_subplot(111)
            cax = ax.matshow(n_cm)
            fig.colorbar(cax)
            ax.set_aspect('auto')
            ax.set_xticks(np.arange(0.0, num_sem_classes, 1.0), minor=False)
            ax.set_yticks(np.arange(0.0, num_sem_classes, 1.0), minor=False)
            ax.set_xticklabels(sem_labels)
            ax.set_yticklabels(sem_labels)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.xaxis.set_ticks_position('bottom')
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            plt.xlabel('Predicted')
            plt.ylabel('Ground truth')
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, f"{int(cm[i, j])}",
                         horizontalalignment="center",
                         color="white" if n_cm[i, j] < 0.5 else "black",
                         fontsize=6)
            plt.tight_layout()
            if save:
                plt.savefig(f"{self.log_path}/confusion_matrix.pdf")
            self.writer.add_figure(name, fig, global_iteration, close=True)

    def log_class_hist(self, name, class_hist, global_iteration, save=False):
        if self.is_empty([class_hist]):
            return
        with torch.no_grad():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = np.array(range(len(class_hist)))
            ax.bar(x, class_hist)
            ax.set_xticks(np.arange(0.0, self.num_semantics, 1.0), minor=False)
            ax.set_xticklabels(self.semantic_labels)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            fig.tight_layout()
            if save:
                plt.savefig(f"{self.log_path}/{name.replace('/', '_')}_{global_iteration}.pdf")
            self.writer.add_figure(name, fig, global_iteration, close=True)

    def log_component_weights(self, name, weights, global_iteration, save):
        if self.is_empty([weights]):
            return
        with torch.no_grad():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = np.array(range(len(weights)))
            ax.bar(x, weights)
            fig.tight_layout()
            if save:
                plt.savefig(f"{self.log_path}/{name.replace('/', '_')}_{global_iteration}.pdf")
            self.writer.add_figure(name, fig, global_iteration, close=True)
