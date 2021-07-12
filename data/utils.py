##############################################################################################################
# Part of the code from
# https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/data/transforms/target_transforms.py
# Improved the original code by offering the option to compute heatmaps across batches and classes jointly
##############################################################################################################
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.ndimage

def get_gaussian(sigma):
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    return torch.Tensor(np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)))

def get_gaussian_filter(sigma, num_semantics, dim):
    n = int(min(sigma + 3, dim - 1))
    n = n if n % 2 == 1 else n - 1
    # n = int(min(6 * sigma + 3, dim - 1))
    assert n % 2 == 1
    x = np.arange(0, n, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = n // 2 + 1, n // 2 + 1
    gaussian = torch.Tensor(np.exp(- ((x - x0) ** 2 + (y - y0) ** 2)))
    kernel = gaussian / torch.sum(gaussian)
    kernel = kernel.view(1, 1, n, n)
    kernel = kernel.repeat(num_semantics, 1, 1, 1)
    gaussian_filter = torch.nn.Conv2d(in_channels=num_semantics, out_channels=num_semantics, kernel_size=gaussian.size(),
                                      groups=num_semantics, bias=False)
    gaussian_filter.weight.data = kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

def get_coord_g(g):
    x_coord_g = torch.ones_like(g)
    y_coord_g = torch.ones_like(g)
    x_coord_g = torch.cumsum(x_coord_g, dim=1) - 1
    y_coord_g = torch.cumsum(y_coord_g, dim=0) - 1
    return x_coord_g, y_coord_g

def get_coord(max_dim, aspect_ratio):
    x_coord = torch.ones((int(max_dim), int(aspect_ratio * max_dim)))
    y_coord = torch.ones((int(max_dim), int(aspect_ratio * max_dim)))
    x_coord = torch.cumsum(x_coord, dim=1) - 1
    y_coord = torch.cumsum(y_coord, dim=0) - 1
    return x_coord, y_coord

def fill_center_heatmap(center, x, y, g, sigma):
    height, width = center.shape[:2]
    upper_left = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
    bottom_right = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
    c, d = max(0, -upper_left[0]), min(bottom_right[0], width) - upper_left[0]
    a, b = max(0, -upper_left[1]), min(bottom_right[1], height) - upper_left[1]
    cc, dd = max(0, upper_left[0]), min(bottom_right[0], width)
    aa, bb = max(0, upper_left[1]), min(bottom_right[1], height)
    center[aa:bb, cc:dd] = torch.max(center[aa:bb, cc:dd], g[a:b, c:d])

def get_center_volume(max_dim, max_sigma, min_sigma):
    center_volume = []
    for d in range(2, int(np.log2(max_dim)) + 1):
        sigma = max(d * max_sigma / max_dim, min_sigma)
        discrete_gaussian = get_gaussian(sigma)
        center_volume.append(torch.sum(discrete_gaussian))
    return center_volume

def get_batch_center_heatmap(height, width, x, y, g, sigma, x_coord, y_coord):
    device = x.get_device()
    height_g, width_g = g.shape
    x_coord_g, y_coord_g = get_coord_g(g)
    batch, channels, num = x.shape
    zero_t = torch.tensor([0]).to(device)
    height_t = torch.tensor([height]).to(device)
    width_t = torch.tensor([width]).to(device)
    up = torch.round(y - 3 * sigma - 1).long()
    left = torch.round(x - 3 * sigma - 1).long()
    bottom = torch.round(y + 3 * sigma + 2).long()
    right = torch.round(x + 3 * sigma + 2).long()
    a = torch.max(zero_t, - up).view(batch, channels, num, 1, 1)
    b = (torch.min(bottom, height_t) - up).view(batch, channels, num, 1, 1)
    c = torch.max(zero_t, - left).view(batch, channels, num, 1, 1)
    d = (torch.min(right, width_t) - left).view(batch, channels, num, 1, 1)
    aa = torch.max(zero_t, up).view(batch, channels, num, 1, 1)
    bb = torch.min(bottom, height_t).view(batch, channels, num, 1, 1)
    cc = torch.max(zero_t, left).view(batch, channels, num, 1, 1)
    dd = torch.min(right, width_t).view(batch, channels, num, 1, 1)
    x_coord = x_coord.expand(batch, channels, num, height, width)
    y_coord = y_coord.expand(batch, channels, num, height, width)
    center_mask = (aa <= y_coord) & (y_coord < bb) & (cc <= x_coord) & (x_coord < dd)
    x_coord_g = x_coord_g.expand(batch, channels, num, height_g, width_g)
    y_coord_g = y_coord_g.expand(batch, channels, num, height_g, width_g)
    g_mask = (a <= y_coord_g) & (y_coord_g < b) & (c <= x_coord_g) & (x_coord_g < d)
    g = g.expand(batch, channels, num, height_g, width_g)
    stacked_center = torch.zeros_like(center_mask).float()
    stacked_center[center_mask] = g[g_mask]
    return torch.max(stacked_center, dim=2)[0]

def get_semantic_features(seg, num_semantics, label_nc, ignored_value=255):
    # extract semantic segmentation map
    seg = seg.clone()
    seg[seg > 999] //=  1000
    if ignored_value is not None and label_nc is not None:
        seg[seg == ignored_value] = label_nc
    # extract semantic conditioning code
    semantic_cond = torch.zeros(num_semantics)
    unique, counts = torch.unique(seg.flatten(), return_counts=True)
    semantic_cond[unique.long()] = counts.float()
    semantic_cond /= torch.sum(semantic_cond)
    # preprocess semantic segmentation map
    seg_mc = torch.zeros([num_semantics, *seg.shape])
    seg_mc = seg_mc.scatter_(0, seg.unsqueeze(0).long(), 1.0)
    return seg_mc, semantic_cond

def get_soft_sem_seg(sem_seg, gaussian_filter, soft_sem_prop):
    sem_seg = sem_seg.unsqueeze(0)
    pad = (gaussian_filter.kernel_size[0] - 1) // 2
    # print("kernel size:", gaussian_filter.kernel_size[0])
    padded_sem_seg = F.pad(sem_seg, (pad, pad, pad, pad), mode='reflect')
    soft_sem_seg = (1 - soft_sem_prop) * sem_seg + soft_sem_prop * gaussian_filter(padded_sem_seg)
    return soft_sem_seg.squeeze(0)

def get_instance_center_offset(seg, g, sigma, scaled_x_coord, scaled_y_coord, height):
    instances = torch.unique(seg[seg > 999])
    instance_center = torch.zeros(seg.shape)
    instance_offset = torch.zeros(tuple([2]) + seg.shape)
    for instance in instances:
        mask = (seg == instance)
        center_x = int(torch.mean(scaled_x_coord[mask]) * height)
        center_y = int(torch.mean(scaled_y_coord[mask]) * height)
        instance_offset[0, mask] = (1. * center_x / height) - scaled_x_coord[mask]
        instance_offset[1, mask] = (1. * center_y / height) - scaled_y_coord[mask]
        fill_center_heatmap(instance_center, center_x, center_y, g, sigma)
    instance_center = instance_center.unsqueeze(0)
    return instance_center, instance_offset

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(centers, sigma, leafsize=10, k=4):
    density = np.zeros(centers.shape, dtype=np.float32)
    centers_count = np.count_nonzero(centers)
    if centers_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(centers)[1], np.nonzero(centers)[0])))
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=k)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(centers.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if not sigma:
            if centers_count == 2:
                sigma = distances[i][1] * 0.3
            elif centers_count == 3:
                sigma = (distances[i][1] + distances[i][2]) * 0.2
            elif centers_count > 3:
                sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            else:
                sigma = np.mean(np.array(centers.shape)) * 0.3
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density

def get_instance_density(seg, things_idx, scaled_x_coord, scaled_y_coord, height, geometric, sigma):
    things_num = len(things_idx)
    instances = torch.unique(seg[seg > 999])
    centers = np.zeros(tuple([things_num]) + seg.shape)
    instance_density = torch.zeros(tuple([things_num]) + seg.shape)
    sigma = sigma if not geometric else None
    for instance in instances:
        idx = things_idx.index(instance // 1000)
        mask = (seg == instance)
        center_x = int(torch.mean(scaled_x_coord[mask]) * height)
        center_y = int(torch.mean(scaled_y_coord[mask]) * height)
        centers[idx, center_y, center_x] = 1
    for idx in range(things_num):
        instance_density[idx] = torch.tensor(gaussian_filter_density(centers[idx], sigma=sigma))
    return instance_density

def get_instance_cond(seg, things_idx):
    things_num = len(things_idx)
    instances = torch.unique(seg[seg > 999])
    instance_cond = torch.zeros(things_num)
    for instance in instances:
        try:
            idx = things_idx.index(instance // 1000)
        except:
            raise ValueError(instance)
        instance_cond[idx] += 1
    return instance_cond

def get_instance_edge(seg, gaussian_filter=None):
    instance_edge = torch.zeros(seg.shape).bool()
    instance_edge[:, 1:] = instance_edge[:, 1:] | (seg[:, 1:] != seg[:, :-1])
    instance_edge[:, :-1] = instance_edge[:, :-1] | (seg[:, 1:] != seg[:, :-1])
    instance_edge[1:, :] = instance_edge[1:, :] | (seg[1:, :] != seg[:-1, :])
    instance_edge[:-1, :] = instance_edge[:-1, :] | (seg[1:, :] != seg[:-1, :])
    instance_edge = instance_edge.float()
    instance_edge = instance_edge.unsqueeze(0)
    if gaussian_filter:
        instance_edge = instance_edge.unsqueeze(0)
        pad = (gaussian_filter.kernel_size[0] - 1) // 2
        instance_edge = F.pad(instance_edge, (pad, pad, pad, pad), mode='reflect')
        instance_edge = gaussian_filter(instance_edge).squeeze(0)
    return instance_edge