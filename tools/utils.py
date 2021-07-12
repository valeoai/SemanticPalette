import os
import PIL
import numpy as np
import scipy.io as sio
from matplotlib import cm
from matplotlib.colors import ListedColormap

import torch
from torchvision import transforms

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dict_to_cuda(dict):
    for k, v in dict.items():
        dict[k] = v.cuda()
    return dict

def dict_to_cpu(dict):
    for k, v in dict.items():
        dict[k] = v.cpu()
    return dict

def dict_detach(dict):
    for k, v in dict.items():
        dict[k] = v.detach()
    return dict

def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = PIL.Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))

def create_colormap(opt):
    print(f"Loading colormat: {opt.colormat}")
    colormat = sio.loadmat('data/colormaps/%s.mat'%opt.colormat)
    return colormat['colors']

def color_transfer(im, colormap):
    im = im.cpu().numpy()
    if np.any(im == -1):
        bg_mask = im == -1
        im += 2
        new_colors = [[125., 125., 125.], [255., 255., 255.]]
        colormap = np.vstack([new_colors, colormap])
        cell = im.shape[2] // 60
        board = np.zeros(im.shape)
        for i in range(cell):
            board[:, :, i::2 * cell] += 1
            board[:, :, :, i::2 * cell] += 1
        board = board % 2
        im[bg_mask] = board[bg_mask]
    im_new = torch.Tensor(im.shape[0], 3, im.shape[2], im.shape[3])
    newcmp = ListedColormap(colormap / 255.0)
    for i in range(im.shape[0]):
        img = (im[i, 0, :, :]).astype('uint8')
        rgba_img = newcmp(img)
        rgb_img = PIL.Image.fromarray((255 * np.delete(rgba_img, 3, 2)).astype('uint8'))
        tt = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        rgb_img = tt(rgb_img)
        im_new[i, :, :, :] = rgb_img
    im_new = im_new
    return im_new

def color_spread(spread, max_spread):
    im_spread = (spread - 1).numpy()
    im_spread[im_spread > 0] = im_spread[im_spread > 0] / max_spread
    im_spread = im_spread / 2 + 0.5
    top = cm.get_cmap('Greys_r', 128)
    bottom = cm.get_cmap('Oranges', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='GreyOrange')
    im_spread_rgba = newcmp(im_spread)
    im_spread_rgb = torch.tensor(im_spread_rgba[:, :, :, :3]).permute(0, 3, 1, 2)
    return im_spread_rgb

# def get_confusion_matrix(sem_index_real, sem_index_pred, num_semantics):
#     index = (sem_index_real * num_semantics + sem_index_pred).long().flatten()
#     count = torch.bincount(index)
#     confusion_matrix = torch.zeros((num_semantics, num_semantics)).to(sem_index_real.get_device())
#     for i_label in range(num_semantics):
#         for j_pred_label in range(num_semantics):
#             cur_index = i_label * num_semantics + j_pred_label
#             if cur_index < len(count):
#                 confusion_matrix[i_label, j_pred_label] = count[cur_index]
#     return confusion_matrix

# def get_confusion_matrix(sem_index_real, sem_index_pred, num_semantics):
#     confusion_matrix = None
#     for real_id, pred_id in zip(sem_index_real, sem_index_pred):

def get_confusion_matrix(sem_index_real, sem_index_pred, num_semantics):
    # bincount is much faster on cpu than gpu
    device = sem_index_real.device
    sem_index_real = sem_index_real.cpu()
    sem_index_pred = sem_index_pred.cpu()
    mask = (sem_index_real >= 0) & (sem_index_real < num_semantics)
    index = (sem_index_real[mask] * num_semantics + sem_index_pred[mask]).long().flatten()
    confusion_matrix = torch.bincount(index, minlength=num_semantics**2).view(num_semantics, num_semantics)
    return confusion_matrix.to(device)

def to_cuda(tensor_dic, key):
    if key in tensor_dic:
        if 0 in tensor_dic[key].size():
            tensor_dic[key] = torch.Tensor([])
        return tensor_dic[key].cuda()
    return torch.Tensor([])

def get_seg_size(num_semantics, num_things, panoptic, instance_type):
    seg_size = num_semantics
    if panoptic:
        if "density" in instance_type:
            seg_size += num_things
        if "center_offset" in instance_type:
            seg_size += 3
        if "edge" in instance_type:
            seg_size += 1
    return seg_size
