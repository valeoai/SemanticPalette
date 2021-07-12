import torch
import torch.nn as nn
import torch.nn.functional as F
from .equalized import EqualizedConv2d
from copy import deepcopy

class BaseActivation(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.out_semantics = opt.num_semantics + 1 if opt.things_stuff else opt.num_semantics
        self.num_things = opt.num_things
        self.panoptic = opt.panoptic
        self.softmax = nn.Softmax2d()
        self.instance_type = opt.instance_type
        self.things_stuff = opt.things_stuff
        self.stuff_idx = opt.stuff_idx
        self.things_idx = opt.things_idx

    def split_ins(self, ins_seg):
        i = 0
        ins_center = torch.Tensor([]).to(ins_seg.get_device())
        ins_offset = torch.Tensor([]).to(ins_seg.get_device())
        ins_edge = torch.Tensor([]).to(ins_seg.get_device())
        ins_density = torch.Tensor([]).to(ins_seg.get_device())
        if self.panoptic:
            if "center_offset" in self.instance_type:
                ins_center = ins_seg[:, i].unsqueeze(1)
                ins_offset = ins_seg[:, i + 1:i + 3]
                i += 3
            if "edge" in self.instance_type:
                ins_edge = ins_seg[:, i].unsqueeze(1)
                i += 1
            if "density" in self.instance_type:
                ins_density = ins_seg[:, i:i + self.num_things]
                i += self.num_things
        return ins_center, ins_offset, ins_edge, ins_density

    def sem_activation(self, sem_seg):
        if self.things_stuff:
            sem_seg[:, self.stuff_idx] = self.softmax(sem_seg[:, self.stuff_idx])
            sem_seg[:, self.things_idx + [-1]] = self.softmax(sem_seg[:, self.things_idx + [-1]])
            sem_seg[:, self.stuff_idx] *= sem_seg[:, [-1]]
            sem_seg = sem_seg[:, :-1]
        else:
            sem_seg = self.softmax(sem_seg)
        return sem_seg

    def ins_activation(self, ins_seg):
        ins_center, ins_offset, ins_edge, ins_density = self.split_ins(ins_seg)
        if self.panoptic:
            ins_center = self.center_activation(ins_center)
            ins_offset = self.offset_activation(ins_offset)
            ins_edge = self.edge_activation(ins_edge)
            ins_density = self.density_activation(ins_density)
        return ins_center, ins_offset, ins_edge, ins_density

    def center_activation(self, ins_center):
        ins_center = torch.sigmoid(ins_center)
        return ins_center

    def offset_activation(self, ins_offset):
        ins_offset = torch.tanh(ins_offset)
        return ins_offset

    def edge_activation(self, ins_edge):
        ins_edge = torch.sigmoid(ins_edge)
        return ins_edge

    def density_activation(self, ins_density):
        ins_density = torch.sigmoid(ins_density)
        return ins_density

    def forward(self, x, cond=None, return_sem_mask=False, as_list=False):
        sem_seg = x[:, :self.out_semantics]
        ins_seg = x[:, self.out_semantics:]

        sem_seg = self.sem_activation(sem_seg)
        ins_center, ins_offset, ins_edge, ins_density = self.ins_activation(ins_seg)

        if as_list:
            x = [sem_seg, ins_center, ins_offset, ins_edge, ins_density]
        else:
            x = torch.cat((sem_seg, ins_center, ins_offset, ins_edge, ins_density), dim=1)

        if return_sem_mask:
            return x, torch.Tensor([])
        else:
            return x


class AssistedActivation(BaseActivation):
    def __init__(self, opt):
        super().__init__(opt)
        self.weakly = "weakly" in opt.cond_mode
        self.cond_seg = opt.cond_seg
        self.sem_assisted = "sem_assisted" in opt.cond_mode
        self.ins_assisted = "ins_assisted" in opt.cond_mode
        self.filter_cond = opt.filter_cond

    def filter_sem_cond(self, sem_cond, h, w):
        if self.filter_cond:
            sem_cond = sem_cond.clone()
            sem_cond[sem_cond < 1. / (h * w)] = 0
            sem_cond /= torch.sum(sem_cond, dim=1, keepdim=True)
        return sem_cond

    def cond_sem_activation(self, sem_seg, sem_cond, eps=0.000001, return_sem_mask=False, force=False):
        if self.things_stuff and not force:
            stuff_cond = sem_cond[:, self.stuff_idx]
            stuff_cond /= torch.sum(stuff_cond, dim=1, keepdim=True)
            things_cond = sem_cond[:, self.things_idx]
            bg_cond = 1. - torch.sum(things_cond, dim=1, keepdim=True)
            things_cond = torch.cat([things_cond, bg_cond], dim=1)
            sem_seg[:, self.stuff_idx], stuff_mask = self.cond_sem_activation(sem_seg[:, self.stuff_idx], stuff_cond,
                                                                              return_sem_mask=True, force=True)
            sem_seg[:, self.things_idx + [-1]], things_mask = self.cond_sem_activation(sem_seg[:, self.things_idx + [-1]], things_cond,
                                                                                      return_sem_mask=True, force=True)
            sem_seg[:, self.stuff_idx] *= sem_seg[:, [-1]]
            sem_seg = sem_seg[:, :-1]
            sem_mask = torch.cat([stuff_mask, things_mask], dim=1)
            return sem_seg, sem_mask

        sem_cond = self.filter_sem_cond(sem_cond, sem_seg.size(2), sem_seg.size(3))
        if self.weakly:
            sem_seg = torch.sigmoid(sem_seg)
        else:
            sem_seg = F.softmax(sem_seg.view(*sem_seg.size()[:2], -1), dim=2).view_as(sem_seg)
        sem_mask = (sem_seg * sem_cond.view(sem_seg.size(0), -1, 1, 1)) * sem_seg.size(2) * sem_seg.size(3)
        sem_seg = (sem_mask + eps) / torch.sum(sem_mask + eps, dim=1, keepdim=True)

        if return_sem_mask:
            return sem_seg, sem_mask
        else:
            return sem_seg

    def cond_ins_activation(self, ins_seg, ins_cond):
        ins_center, ins_offset, ins_edge, ins_density = self.split_ins(ins_seg)
        if self.panoptic:
            ins_center = self.center_activation(ins_center)
            ins_offset = self.offset_activation(ins_offset)
            ins_edge = self.edge_activation(ins_edge)
            ins_density = self.cond_density_activation(ins_density, ins_cond)
        return ins_center, ins_offset, ins_edge, ins_density

    def cond_density_activation(self, ins_density, ins_cond):
        ins_density = F.softmax(ins_density.view(*ins_density.size()[:2], -1), dim=2).view_as(ins_density)
        ins_density = ins_density * ins_cond.view(ins_density.size(0), -1, 1, 1)
        return ins_density

    def forward(self, x, cond=None, return_sem_mask=False, as_list=False):
        sem_cond = cond["sem_cond"]
        ins_cond = cond["ins_cond"]

        sem_seg = x[:, :self.out_semantics]
        ins_seg = x[:, self.out_semantics:]

        if self.sem_assisted:
            sem_seg, sem_mask = self.cond_sem_activation(sem_seg, sem_cond, return_sem_mask=True)
        else:
            sem_seg = self.sem_activation(sem_seg)
            sem_mask = torch.Tensor([])

        if self.ins_assisted:
            ins_center, ins_offset, ins_edge, ins_density = self.cond_ins_activation(ins_seg, ins_cond)
        else:
            ins_center, ins_offset, ins_edge, ins_density = self.ins_activation(ins_seg)

        if as_list:
            x = [sem_seg, ins_center, ins_offset, ins_edge, ins_density]
        else:
            x = torch.cat((sem_seg, ins_center, ins_offset, ins_edge, ins_density), dim=1)

        if return_sem_mask:
            return x, sem_mask
        else:
            return x


class AssistedBlock(nn.Module):
    def __init__(self, in_channels, opt, to_mask=None):
        super().__init__()
        self.num_semantics = opt.num_semantics
        self.cond_activation = AssistedActivation(opt)
        self.to_mask = to_mask if to_mask else Mask(in_channels, opt)
        hidden_size = opt.num_semantics
        if opt.panoptic:
            if "density" in opt.instance_type:
                hidden_size += opt.num_things
            if "center_offset" in opt.instance_type:
                hidden_size += 3
            if "edge" in opt.instance_type:
                hidden_size += 1
        mul = 2 if opt.joint_type == "affine" else 1
        self.in_channels = in_channels
        self.conv1 = EqualizedConv2d(hidden_size, in_channels * mul, 3, padding=1, gain=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = torch.sigmoid
        self.joint_type = opt.joint_type

    def forward(self, x, cond=None):
        raw = self.to_mask(x)
        raw_sem_seg = raw[:, :self.num_semantics]
        seg, sem_mask = self.cond_activation(raw, cond, return_sem_mask=True, as_list=True)
        sem_seg, ins_center, ins_offset, ins_edge, ins_density = seg
        seg = torch.cat(seg, dim=1)
        y = self.conv1(seg)
        z = torch.tensor([0]).float().to(x.get_device())
        if self.joint_type == "affine":
            alpha, beta = y[:, :self.in_channels], y[:, self.in_channels:]
        elif self.joint_type == "bias":
            alpha, beta = z, y
        elif self.joint_type == "linear":
            alpha, beta = y, z
        x = self.sigmoid(alpha) * x + self.lrelu(beta)

        output_dict = {"sem_seg": sem_seg,
                       "ins_center": ins_center,
                       "ins_offset": ins_offset,
                       "ins_edge": ins_edge,
                       "ins_density": ins_density,
                       "sem_mask": sem_mask,
                       "raw_sem_seg": raw_sem_seg}

        return x, output_dict


class Mask(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.panoptic = opt.panoptic
        out_semantics = opt.num_semantics + 1 if opt.things_stuff else opt.num_semantics
        self.sem_conv = EqualizedConv2d(in_channels, out_semantics, 1, gain=1)
        if self.panoptic:
            out_channels = 0
            if "density" in opt.instance_type:
                out_channels += opt.num_things
            if "center_offset" in opt.instance_type:
                out_channels += 3
            if "edge" in opt.instance_type:
                out_channels += 1
            self.ins_conv = EqualizedConv2d(in_channels, out_channels, 1, gain=1)

    def forward(self, x):
        sem_seg = self.sem_conv(x)
        if self.panoptic:
            ins_seg = self.ins_conv(x)
            x = torch.cat((sem_seg, ins_seg), dim=1)
        else:
            x = sem_seg
        return x


class CondJoint(nn.Module):
    def __init__(self, in_channels, opt, joints_mul=0):
        super().__init__()
        self.sem_assisted = "sem_assisted" in opt.cond_mode
        self.ins_assisted = "ins_assisted" in opt.cond_mode
        self.num_semantics = opt.num_semantics

        self.to_mask = Mask(in_channels, opt)

        self.assisted_blocks = []
        if self.sem_assisted or self.ins_assisted:
            self.cond_activation = AssistedActivation(opt)
            for i in range(joints_mul):
                self.assisted_blocks.append(AssistedBlock(in_channels, opt, to_mask=self.to_mask))
            self.assisted_blocks = nn.ModuleList(self.assisted_blocks)
        else:
            self.base_activation = BaseActivation(opt)

    def forward(self, x, cond=None):
        segs = []
        for block in self.assisted_blocks:
            x, seg = block(x, cond)
            segs.append(seg)
        return x, segs

    def mask(self, x, cond=None):
        if self.sem_assisted or self.ins_assisted:
            raw = self.to_mask(x)
            x, sem_mask = self.cond_activation(raw, cond, return_sem_mask=True, as_list=True)
            raw_sem_seg = raw[:, :self.num_semantics]
        else:
            x = self.to_mask(x)
            x = self.base_activation(x, as_list=True)
            sem_mask = torch.tensor([])
            raw_sem_seg = torch.tensor([])
        sem_seg, ins_center, ins_offset, ins_edge, ins_density = x
        output_dict = {"sem_seg": sem_seg,
                       "ins_center": ins_center,
                       "ins_offset": ins_offset,
                       "ins_edge": ins_edge,
                       "ins_density": ins_density,
                       "sem_mask": sem_mask,
                       "raw_sem_seg": raw_sem_seg}
        return output_dict


class CondMask(nn.Module):
    def __init__(self, in_channels, opt, use_bg=False):
        super().__init__()
        opt = deepcopy(opt)
        if use_bg:
            opt.num_semantics += 1
        self.sem_assisted = "sem_assisted" in opt.cond_mode
        self.ins_assisted = "ins_assisted" in opt.cond_mode
        self.num_semantics = opt.num_semantics

        self.to_mask = Mask(in_channels, opt)

        if self.sem_assisted or self.ins_assisted:
            self.cond_activation = AssistedActivation(opt)
        else:
            self.base_activation = BaseActivation(opt)

    def forward(self, x, cond=None):
        if self.sem_assisted or self.ins_assisted:
            raw = self.to_mask(x)
            x, sem_mask = self.cond_activation(raw, cond, return_sem_mask=True, as_list=True)
            raw_sem_seg = raw[:, :self.num_semantics]
        else:
            x = self.to_mask(x)
            x = self.base_activation(x, as_list=True)
            sem_mask = torch.tensor([])
            raw_sem_seg = torch.tensor([])
        sem_seg, ins_center, ins_offset, ins_edge, ins_density = x
        output_dict = {"sem_seg": sem_seg,
                       "ins_center": ins_center,
                       "ins_offset": ins_offset,
                       "ins_edge": ins_edge,
                       "ins_density": ins_density,
                       "sem_mask": sem_mask,
                       "raw_sem_seg": raw_sem_seg}
        return output_dict