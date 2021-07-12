"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
from . import utils
import PIL
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch
import scipy.io as sio
from matplotlib.colors import ListedColormap
from torchvision import transforms
import numpy as np
import datetime

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.colormap = self.create_colormap(opt)
        if self.tf_log:
            # import tensorflow as tf
            # self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs', datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
            # self.writer = tf.summary.create_file_writer(self.log_dir)
            # tf.summary.FileWriter(self.log_dir)
            self.writer = SummaryWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            utils.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def create_colormap(self, opt):
        colormap = None
        if 'indoor' in opt.dataset_mode:
            colormat = sio.loadmat('../SBGAN/datasets/color_indoor.mat')
            colormap = colormat['colors']
        elif 'cityscapes' in opt.dataset_mode:
            colormat = sio.loadmat('../SBGAN/datasets/cityscapes_color35.mat')
            colormap = colormat['colors']
        return colormap

    def color_transfer(self, im):
        im = im.cpu().numpy()
        im_new = torch.Tensor(im.shape[0], 3, im.shape[2], im.shape[3])
        newcmp = ListedColormap(self.colormap / 255.0)
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

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        if self.tf_log:
            for key, t in visuals.items():
                if t.shape[1] == 1:
                    t = self.color_transfer(t)
                grid = make_grid(t, nrow=4, normalize=True, range=(-1, 1))
                self.writer.add_image(key, grid, step)

        # ## convert tensors to numpy arrays
        # visuals = self.convert_visuals_to_numpy(visuals)
        #
        # if self.tf_log: # show images in tensorboard output
        #     img_summaries = []
        #     for label, image_numpy in visuals.items():
        #         # Write the image to a string
        #         try:
        #             s = StringIO()
        #         except:
        #             s = BytesIO()
        #         if len(image_numpy.shape) >= 4:
        #             image_numpy = image_numpy[0]
        #         PIL.Image.fromarray(image_numpy).save(s, format="jpeg")
        #         # Create an Image object
        #         img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
        #         # Create a Summary value
        #         img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))
        #
        #     # Create and write Summary
        #     summary = self.tf.Summary(value=img_summaries)
        #     self.writer.add_summary(summary, step)
        #
        # if self.use_html: # save images to a html file
        #     for label, image_numpy in visuals.items():
        #         if isinstance(image_numpy, list):
        #             for i in range(len(image_numpy)):
        #                 img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s_%d.png' % (epoch, step, label, i))
        #                 utils.save_image(image_numpy[i], img_path)
        #         else:
        #             img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s.png' % (epoch, step, label))
        #             if len(image_numpy.shape) >= 4:
        #                 image_numpy = image_numpy[0]
        #             utils.save_image(image_numpy, img_path)
        #
        #     # update website
        #     webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
        #     for n in range(epoch, 0, -1):
        #         webpage.add_header('epoch [%d]' % n)
        #         ims = []
        #         txts = []
        #         links = []
        #
        #         for label, image_numpy in visuals.items():
        #             if isinstance(image_numpy, list):
        #                 for i in range(len(image_numpy)):
        #                     img_path = 'epoch%.3d_iter%.3d_%s_%d.png' % (n, step, label, i)
        #                     ims.append(img_path)
        #                     txts.append(label+str(i))
        #                     links.append(img_path)
        #             else:
        #                 img_path = 'epoch%.3d_iter%.3d_%s.png' % (n, step, label)
        #                 ims.append(img_path)
        #                 txts.append(label)
        #                 links.append(img_path)
        #         if len(ims) < 10:
        #             webpage.add_images(ims, txts, links, width=self.win_size)
        #         else:
        #             num = int(round(len(ims)/2.0))
        #             webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
        #             webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
        #     webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                self.writer.add_scalar(tag, value, step)
                # with self.writer.as_default():
                #     self.tf.summary.scalar(tag, value)
                #     self.writer.flush()
                # summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                # self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                t = utils.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = utils.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            utils.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
