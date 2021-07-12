import numpy as np

import torch
import torch.nn.functional as F

from data.utils import get_gaussian, get_coord, fill_center_heatmap, get_batch_center_heatmap

class InstanceRefiner:
    def __init__(self, opt):
        self.num_semantics = opt.num_semantics
        self.num_things = opt.num_things
        self.things_idx = torch.Tensor(opt.things_idx).long()
        self.center_thresh = opt.center_thresh
        dim = [2 ** k for k in range(2, int(np.log2(opt.max_dim)) + 1)]
        self.sigma = [max(d * opt.max_sigma / opt.max_dim, opt.min_sigma) for d in dim]
        self.g = [get_gaussian(sigma) for sigma in self.sigma]
        self.x_coord, self.y_coord = get_coord(opt.max_dim, opt.aspect_ratio)
        self.dim_ind = int(np.log2(opt.seg_dim)) - 2

    def is_empty(self, tensors):
        for tensor in tensors:
            if tensor.size(0) == 0:
                return True
        return False

    def filter_offset(self, ins_offset, seg_mc):
        if self.is_empty([ins_offset, seg_mc]):
            return ins_offset
        things_seg = seg_mc.detach()[:, self.things_idx]
        has_thing_in_pxl = torch.sum(things_seg, dim=1, keepdim=True) > 0
        ins_offset = ins_offset * has_thing_in_pxl
        return ins_offset

    def filter_density(self, ins_density, seg_mc):
        if self.is_empty([ins_density, seg_mc]):
            return ins_density
        things_seg = seg_mc.detach()[:, self.things_idx]
        has_thing_in_image = torch.sum(things_seg, dim=(2, 3), keepdim=True) > 0
        ins_density = ins_density * has_thing_in_image
        return ins_density

    def transform(self, ins_center, ins_offset, seg_mc):
        if self.is_empty([ins_center, ins_offset, seg_mc]):
            return ins_center, ins_offset
        bs = ins_center.shape[0]
        device = ins_center.get_device()
        height, width = ins_center.shape[2:]
        things_seg = seg_mc.detach()[:, self.things_idx]
        has_thing_in_image = torch.sum(things_seg, dim=(2, 3), keepdim=True) > 0
        nms_mask = self.get_nms_mask(ins_center, self.dim_ind)
        thresh_mask = self.get_threshold_mask(ins_center)
        pseudo_center = torch.zeros_like(ins_center)
        pseudo_offset = torch.zeros_like(ins_offset)
        scaled_x_coord = (self.x_coord[:height, :width] / height).to(device)
        scaled_y_coord = (self.y_coord[:height, :width] / height).to(device)
        g = self.g[self.dim_ind].to(device)
        sigma = self.sigma[self.dim_ind]
        fake_center = nms_mask & thresh_mask
        fake_center_count = torch.sum(fake_center.float(), dim=(2, 3))
        for i in range(bs):
            for j in range(self.num_things):
                if has_thing_in_image[i, j]:
                    mask = (things_seg[i, j] > 0)
                    mask_shape = torch.sum(mask.long())
                    if fake_center_count[i, j] <= 1:
                        pseudo_x = int(torch.mean(scaled_x_coord[mask]) * height)
                        pseudo_y = int(torch.mean(scaled_y_coord[mask]) * height)
                        pseudo_offset[i, 0, mask] = (1. * pseudo_x / height) - scaled_x_coord[mask]
                        pseudo_offset[i, 1, mask] = (1. * pseudo_y / height) - scaled_y_coord[mask]
                        fill_center_heatmap(pseudo_center[i, j], pseudo_x, pseudo_y, g, sigma)
                    else:
                        closest_center = -1 * torch.ones(mask_shape, dtype=torch.long).to(device)
                        closest_dis = 10000 * torch.ones(mask_shape).to(device)
                        for k, (fake_y, fake_x) in enumerate(fake_center[i, j].nonzero()):
                            dis_x = scaled_x_coord[mask] - 1. * fake_x / height
                            dis_y = scaled_y_coord[mask] - 1. * fake_y / height
                            squared_dis_to_center = dis_x ** 2 + dis_y ** 2
                            new_center_mask = (squared_dis_to_center < closest_dis)
                            closest_dis[new_center_mask] = squared_dis_to_center[new_center_mask]
                            closest_center[new_center_mask] = k
                        for k in range(int(fake_center_count[i, j])):
                            sub_mask = closest_center == k
                            if torch.sum(sub_mask.float()) > 0:
                                pseudo_x = int(torch.mean(scaled_x_coord[mask][sub_mask]) * height)
                                pseudo_y = int(torch.mean(scaled_y_coord[mask][sub_mask]) * height)
                                pseudo_offset[i, 0, mask][sub_mask] = (pseudo_x / height) \
                                                                      - scaled_x_coord[mask][sub_mask]
                                pseudo_offset[i, 1, mask][sub_mask] = (pseudo_y / height) \
                                                                      - scaled_y_coord[mask][sub_mask]
                                fill_center_heatmap(pseudo_center[i, j], pseudo_x, pseudo_y, g, sigma)
        return pseudo_center, pseudo_offset

    def batch_transform(self, ins_center, ins_offset, seg_mc, num=20):
        # prepare data
        batch, channels, height, width = ins_center.shape
        device = ins_center.get_device()
        things_seg = seg_mc.detach()[:, self.things_idx]
        things_id = things_seg.max(1, keepdim=True)[1]
        nms_mask = self.get_nms_mask(ins_center, self.dim_ind)
        thresh_mask = self.get_threshold_mask(ins_center)
        x_coord = self.x_coord[:height, :width].to(device)
        y_coord = self.y_coord[:height, :width].to(device)
        scaled_x_coord = (self.x_coord[:height, :width] / height).to(device)
        scaled_y_coord = (self.y_coord[:height, :width] / height).to(device)
        g = self.g[self.dim_ind].to(device)
        sigma = self.sigma[self.dim_ind]
        # count centers
        is_center = nms_mask & thresh_mask
        center_count = torch.sum(is_center.float(), dim=(2, 3))
        thing_count = torch.sum(things_seg, dim=(2, 3))
        # if there is no center and class is in image take highest value(s) as center(s)
        no_center = (center_count == 0) & (thing_count > 0)
        is_center[no_center] = nms_mask[no_center] == torch.max(nms_mask.view(batch, channels, -1), dim=2)[0].view(batch, channels, 1, 1)[no_center]
        # filter top k centers for each class
        center_values, center_pos = torch.topk((ins_center * is_center).view(batch, channels, -1), num, dim=2)
        fake_x = center_pos % width
        fake_y = center_pos / width
        # put points which are not genuine centers far away
        fake_x[center_values == 0] = - 10000
        fake_y[center_values == 0] = - 10000
        # expand to spatial dim
        fake_x = fake_x.expand(height, width, batch, channels, num).permute(2, 3, 4, 0, 1) # batch, channels, num, height, width
        fake_y = fake_y.expand(height, width, batch, channels, num).permute(2, 3, 4, 0, 1) # batch, channels, num, height, width
        # find closest center
        things_id = things_id.expand(num, batch, 1, height, width).permute(1, 2, 0, 3, 4) # batch, 1, num, height, width
        dis_x = 1. * torch.gather(fake_x, 1, things_id)[:, 0] / height - (ins_offset[:, 0].unsqueeze(1) + scaled_x_coord.view(1, 1, height, width))
        dis_y = 1. * torch.gather(fake_y, 1, things_id)[:, 0] / height - (ins_offset[:, 1].unsqueeze(1) + scaled_y_coord.view(1, 1, height, width))
        squared_dis = dis_x ** 2 + dis_y ** 2

        closest_center = torch.min(squared_dis, dim=1, keepdim=True)[1]
        closest_center_mc = torch.zeros(batch, num, height, width).to(device)
        closest_center_mc.scatter_(1, closest_center, 1.0)
        closest_center_mc = closest_center_mc.view(batch, 1, num, height, width)
        # compute refined centers (pseudo center) based on attributions
        pseudo_instance = things_seg.view(batch, channels, 1, height, width) * closest_center_mc
        pseudo_instance_x = pseudo_instance * scaled_x_coord.view(1, 1, 1, height, width)
        pseudo_instance_y = pseudo_instance * scaled_y_coord.view(1, 1, 1, height, width)
        sum_pseudo_instance = torch.sum(pseudo_instance, dim=(3, 4))
        unpopular_instance = sum_pseudo_instance == 0
        sum_pseudo_instance[unpopular_instance] = 1
        pseudo_x = torch.round(1. * torch.sum(pseudo_instance_x, dim=(3, 4)) / sum_pseudo_instance * height).long()
        pseudo_x[unpopular_instance] = - 10000
        pseudo_y = torch.round(1. * torch.sum(pseudo_instance_y, dim=(3, 4)) / sum_pseudo_instance * height).long()
        pseudo_y[unpopular_instance] = - 10000
        pseudo_center = get_batch_center_heatmap(height, width, pseudo_x, pseudo_y, g, sigma, x_coord, y_coord)
        # compute refined offsets corresponding to pseudo center
        pseudo_instance_offset_x = 1. * pseudo_x.view(batch, channels, num, 1, 1) / height - pseudo_instance_x
        pseudo_instance_offset_y = 1. * pseudo_y.view(batch, channels, num, 1, 1) / height - pseudo_instance_y
        pseudo_offset_x = pseudo_instance_offset_x * pseudo_instance
        pseudo_offset_y = pseudo_instance_offset_y * pseudo_instance
        pseudo_offset = torch.empty_like(ins_offset)
        pseudo_offset[:, 0] = torch.sum(pseudo_offset_x, dim=(1, 2))
        pseudo_offset[:, 1] = torch.sum(pseudo_offset_y, dim=(1, 2))
        return pseudo_center, pseudo_offset

    def get_nms_mask(self, ins_center):
        width = int(self.sigma[self.dim_ind])
        width = width + 1 if width % 2 == 0 else width
        flat_center = ins_center.squeeze(1)
        window_maxima = F.max_pool2d(flat_center, kernel_size=width, stride=1, padding=width // 2).view_as(ins_center)
        return (window_maxima == ins_center).bool()

    def get_threshold_mask(self, ins_center):
        return (ins_center > self.center_thresh).bool()

    def get_peak_mask(self, ins_center):
        if self.is_empty([ins_center]):
            return ins_center
        return self.get_nms_mask(ins_center) & self.get_threshold_mask(ins_center)

    def clean_ins_offset(self, ins_offset, seg_mc):
        if ins_offset.size(0) > 0:
            shape = ins_offset.shape
            ins_offset = ins_offset.view(shape[0], -1, 2, shape[-2], shape[-1])
            seg_mc = seg_mc.view(shape[0], -1, 1, shape[-2], shape[-1])
            ins_offset = ins_offset * seg_mc.detach()[:, self.things_idx]
            ins_offset = ins_offset.view(shape[0], -1, shape[-2], shape[-1])
        return ins_offset