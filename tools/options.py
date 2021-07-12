import os
import sys
import torch
import pickle
import datetime
import argparse
from argparse import Namespace

from tools import utils


SEM_CITYSCAPES = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground',
                  'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
                  'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                  'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle',
                  'license plate']

SEM_IDD = ['road', 'parking', 'drivable fallback', 'sidewalk', 'rail track', 'non-drivable fallback', 'person',
           'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'caravan', 'trailer',
           'train', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign',
           'traffic light', 'pole', 'polegroup', 'obs-str-bar-fallback', 'building', 'bridge' , 'tunnel', 'vegetation',
           'sky', 'fallback background','unlabeled', 'ego vehicle', 'rectification border', 'out of roi',
           'license plate']

SEM_CELEBA = ['null', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
              'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

SEM_ADE = [str(i) for i in range(95)]

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser = self.initialize_base(parser)
        parser = self.initialize_seg_generator(parser)
        parser = self.initialize_img_generator(parser)
        parser = self.initialize_segmentor(parser)
        parser = self.initialize_extra_dataset(parser)
        self.initialized = True
        return parser

    def initialize_base(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='my_experiment', help='name of the experiment, it indicates where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # for mixed precision
        parser.add_argument('--use_amp', action='store_true', help='if specified, use apex mixed precision')
        parser.add_argument('--amp_level', type=str, default='O1', help='O1, O2...')

        # for input / output sizes
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--true_dim', type=int, default=1024, help='resolution of saved images')
        parser.add_argument('--max_dim', type=int, default=512, help='resolution up to which we wish to train our models')
        parser.add_argument('--dim', type=int, default=-1, help='resolution at which to initialize training (has no effect for the seg generator)')
        parser.add_argument('--seg_dim', type=int, default=-1, help='resolution at which to generate segmentation (they are then resized to dim)')
        parser.add_argument('--force_seg_dim', action='store_true', help='if True, load seg at seg_dim')
        parser.add_argument('--bilimax', action='store_true', help='if True, apply bilinear upsampling to seg then max discretizer')
        parser.add_argument('--true_ratio', type=float, default=1.0, help='ratio width/height of saved images, final width will be max_dim * aspect_ratio')
        parser.add_argument('--aspect_ratio', type=float, default=2.0, help='target width/height ratio')
        parser.add_argument('--num_semantics', type=int, default=3, help='number of semantic classes including eventual unknown class')
        parser.add_argument('--semantic_labels', type=str, default=[], nargs="+", help='name of the semantic class for each index')
        parser.add_argument('--label_nc', type=int, default=None, help='new label for unknown class if there is any')
        parser.add_argument('--not_sort', action='store_true', help='if specified, do *not* sort the input paths')
        parser.add_argument('--soft_sem_seg', action='store_true', help='apply gaussian blur to semantic segmentation')
        parser.add_argument('--soft_sem_prop', type=float, default=0.5, help='amount of final sem map with blur')
        parser.add_argument('--transpose', action='store_true', help='transpose the input seg/img')
        parser.add_argument('--imagenet_norm', action='store_true', help='normalize images the same way as it is done for imagenet')
        parser.add_argument('--colorjitter', action='store_true', help='randomly change the brightness, contrast and saturation of images')

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
        parser.add_argument('--dataset', type=str, default='cityscapes')
        parser.add_argument('--load_extra', action='store_true', help='if true, load extended version of dataset if available')
        parser.add_argument('--load_minimal_info', action='store_true', help='if true, load extended version of dataset if available')
        parser.add_argument('--data_idx_type', type=str, default='both', help='(even | odd | both)')
        parser.add_argument('--data_city_type', type=str, default='both', help='(a | no_a | both)')
        parser.add_argument('--has_tgt', action='store_true', help='if false, tgt cond overrides true cond')
        parser.add_argument('--estimated_cond', action='store_true', help='if true, teach a model to generate cond and sample from it')
        parser.add_argument('--nearest_cond_index', action='store_true', help='if true, sample data points which corresponds to the nearest cond')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_h_flip', action='store_true', help='if specified, do not horizontally flip the images for data argumentation')
        parser.add_argument('--no_v_flip', action='store_true', help='if specified, do not vertically flip the images for data argumentation')
        parser.add_argument('--resize_img', type=int, nargs="+", default=None, help='if specified, resize images once they are loaded')
        parser.add_argument('--resize_seg', type=int, nargs="+", default=None, help='if specified, resize segmentations once they are loaded')
        parser.add_argument('--min_zoom', type=float, default=1., help='parameter for augmentation method consisting in zooming and cropping')
        parser.add_argument('--max_zoom', type=float, default=1., help='parameter for augmentation method consisting in zooming and cropping')
        parser.add_argument('--fixed_crop', type=int, nargs="+", default=None, help='if specified, apply a random crop of the given size')
        parser.add_argument('--fixed_top_centered_zoom', type=float, default=None, help='if specified, crop the image to the upper center part')
        parser.add_argument('--num_workers', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='maximum # of samples allowed per dataset, if the dataset directory contains more than max_dataset_size, only a subset is loaded')
        parser.add_argument('--load_from_opt_file', action='store_true', help='loads the options_spade from checkpoints and use that as default')
        parser.add_argument('--no_pairing_check', action='store_true', help='if specified, skip sanity check of correct label-image file pairing')

        # for panoptic mode
        parser.add_argument('--load_panoptic', action='store_true', help='if true, loads both instance and semantic information from segmentation maps, otherwise only semantic information')
        parser.add_argument('--instance_type', type=str, default='center_offset', help='combination of (center_offset | (soft_)edge | density)')
        parser.add_argument('--things_idx', type=int, nargs="+", default=[], help='indexes corresponding to things (by opposition to stuff)')
        parser.add_argument('--max_sigma', type=float, default=8., help='sigma of 2d gaussian representing instance centers for max dim')
        parser.add_argument('--min_sigma', type=float, default=2., help='sigmaiiii of 2d gaussian representing instance centers for min dim')
        parser.add_argument('--center_thresh', type=float, default=0.5, help='threshold to filter instance centers')

        # for display and checkpointing
        parser.add_argument('--log_freq', type=int, default=100, help='frequency at which logger is updated with images')
        parser.add_argument('--save_freq', type=int, default=-1, help='frequency of saving models, if -1 don\'t save')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest model')
        parser.add_argument('--save_path', type=str, default='./')
        parser.add_argument('--colormat', type=str, default='', help='name of colormat to display semantic maps')

        # for training
        parser.add_argument('--niter', type=int, default=1000, help='number of training iterations')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--iter_function', type=str, default=None, help='(iter | cycle)')

        # for testing
        parser.add_argument('--nums_fid', type=int, default=100, help='number of samples to generate to compute fid score')
        parser.add_argument('--slide_eval', action='store_true', help='if true, eval on sliding window')
        parser.add_argument('--multi_scale_eval', action='store_true', help='if true, eval on two scales')
        parser.add_argument('--eval_batchsize', type=int, default=16, help='batch size to compute fid')
        parser.add_argument('--eval_freq', type=int, default=10, help='frequency for evaluting fid')
        parser.add_argument('--no_eval', action='store_true', help='if true, dont do eval')
        parser.add_argument('--eval_idx', type=int, nargs="+", default=[], help="selected classes for evaluation")
        parser.add_argument('--force_eval_batch_size', type=int, default=None, help='if true, force eval batch size for segmentor')

        # for engine
        parser.add_argument('--local_rank', type=int, default=0, help='process rank on node')

        # for sampler
        parser.add_argument('--sampler_weights_method', type=str, default=None, help='(highlight-)(linear | exponential)')
        parser.add_argument('--sampler_bias_method', type=str, default=None, help='(highlight-)linear')
        parser.add_argument('--sampler_weights_scale', type=float, default=2., help='rescale sampling weights to range [0, sampler_scale]')
        parser.add_argument('--sampler_bias_mul', type=float, default=1., help='amplify std for classes that we wish to bias')
        parser.add_argument('--sampler_method', type=str, default="", help='(weights-bias | weights | bias)')

        # for estimator
        parser.add_argument('--estimator_load_path', type=str, default=None, help='load an estimator model from specified folder')
        parser.add_argument('--estimator_min_components', type=int, default=1, help='min number of components for gmm model')
        parser.add_argument('--estimator_max_components', type=int, default=5, help='max number of components for gmm model')
        parser.add_argument('--estimator_force_components', type=int, default=None, help='if not None, fix number of components for gmm model (overrides min and max)')
        parser.add_argument('--estimator_n_init', type=int, default=1, help='number of initializations for gmm model')
        parser.add_argument('--estimator_iter_data', type=int, default=1, help='number of time to iter through data to extract cond codes')
        parser.add_argument('--estimator_projection_mode', type=str, default="approx", help='(approx | iter)')
        parser.add_argument('--estimator_force_bias', type=int, nargs="+", default=[], help="force bias to 1 for specified classes")
        parser.add_argument('--estimator_filter_idx', type=int, nargs="+", default=[], help="prevent sem classes at given idx from being sampled")
        parser.add_argument('--estimator_force_min_class_p', type=float, nargs="+", default=[], help="pair of (class, p) values where surface proportion should be at least p for class")

        # for nearest cond indexor
        parser.add_argument('--indexor_load_path', type=str, default=None, help='load an indexor model from specified folder')
        parser.add_argument('--indexor_normalize', action='store_true', help='if true, in indexor input classes are normalized individually')

        # for end-to-end
        parser.add_argument('--fake_from_fake_dis', type=str, default="both", help='(d | d2 | both)')
        parser.add_argument('--fake_from_real_dis', type=str, default="both", help='(d | d2 | both)')
        parser.add_argument('--img_for_d_real', type=str, default="source", help='(source | target | both)')
        parser.add_argument('--img_for_d_fake', type=str, default="target", help='(source | target | both)')
        parser.add_argument('--img_for_d2_real', type=str, default="target", help='(source | target | both)')
        parser.add_argument('--img_for_d2_fake', type=str, default="target", help='(source | target | both)')
        parser.add_argument('--sem_only_real', action='store_true', help='if true, compute only semantic alignement for real data')
        parser.add_argument('--lambda_d2_from_real', type=float, default=1, help='parameter for second discriminator and fake data')
        parser.add_argument('--no_update_seg_model', action='store_true', help='if true, dont update seg model in end-to-end configuration')
        parser.add_argument('--eval_dataset', type=str, default="base", help='(base | extra)')

        # for offline generation
        parser.add_argument('--save_data_path', type=str, default="datasets/cityscapes_synthetic", help='folder in which to store synthetic data')
        parser.add_argument('--data_num', type=int, default=2975, help="number of synthetic pairs to generate")
        parser.add_argument('--save8bit', action='store_true', help='if true, save semantic segmentation in 8 bit format')

        # for visualizer
        parser.add_argument('--vis_method', type=str, default="", help='method for visualization')
        parser.add_argument('--vis_steps', type=int, default=32, help='method for visualization')
        parser.add_argument('--vis_dataloader_bs', type=int, default=1, help='batch size for dataloader')
        parser.add_argument('--extraction_path', type=str, default=None, help="folder containing mean style codes")
        parser.add_argument('--mean_style_only', action='store_true', help='if true, do not recompute style from image')
        parser.add_argument('--addition_mode', action='store_true', help='if true, shape target for partial edition rather than full')
        parser.add_argument('--save_full_res', action='store_true', help='if true, save as individual images at full resolution')
        parser.add_argument('--vis_ins', action='store_true', help='if true, visualize instance related masks')
        parser.add_argument('--vis_random_style', action='store_true', help='if true, load random style instead of mean style for new elements')
        # for offline generator


        return parser

    def initialize_seg_generator(self, parser):
        # for model
        parser.add_argument('--s_model', type=str, default='progressive', help='(progressive | style)')
        parser.add_argument('--s_seg_type', type=str, default='generator', help='(generator | completor)')
        parser.add_argument('--s_panoptic', action='store_true', help='if true, panoptic segmentation generation, otherwise semantic segmentation generation')
        parser.add_argument('--s_latent_dim', type=int, default=512, help='dimension of the latent vector')
        parser.add_argument('--s_max_hidden_dim', type=int, default=512, help='maximum number of hidden feature maps')
        parser.add_argument('--s_discretization', type=str, default='gumbel', help='(gumbel | max)')

        # for conditional generation
        parser.add_argument('--s_cond_seg', type=str, default=None, help='(semantic | instance | panoptic | None)')
        parser.add_argument('--s_joints_mul', type=int, default=0, help='number of assisted joints between generator blocks to refine intermediate outputs')
        parser.add_argument('--s_joint_type', type=str, default="bias", help='(linear | bias | affine)')
        parser.add_argument('--s_cond_mode', default='sem_recover', help='(entropy &| sem_recover &| (weakly_)assisted &| spread &| ins_recover | original_cgan)')
        parser.add_argument('--s_filter_cond', action='store_true', help='if specified, sem should represent at least one pixel to be taken into account in assisted activation')
        parser.add_argument('--s_pseudo_supervision', action='store_true', help='self supervision for instance related output')
        parser.add_argument('--s_lambda_things', type=float, default=1., help='parameter for things related loss')
        parser.add_argument('--s_lambda_stuff', type=float, default=1., help='parameter for stuff related loss')
        parser.add_argument('--s_lambda_adv_things', type=float, default=1., help='parameter for things gen/dis loss')
        parser.add_argument('--s_things_dis', action='store_true', help='if specified, do an extra forward pass in discriminator with things alone')
        parser.add_argument('--s_ova_idx', type=int, nargs="+", default=[], help='indices for which we wish to apply the one-versus-all loss')
        parser.add_argument('--s_lambda_ova', type=float, default=1., help='parameter for ova loss')
        parser.add_argument('--s_lambda_spread', type=float, default=1., help='parameter for spread loss')

        # for for input / output sizes
        parser.add_argument('--s_things_stuff', action='store_true', help='if specified, treats things and stuff separately')
        parser.add_argument('--s_override_num_semantics', type=int, default=None, help='if not None, overrides num semantics')
        parser.add_argument('--s_sem_conv', type=int, nargs="+", default=None, help='convert seg classes for img generator')

        # for training
        parser.add_argument('--s_optimizer', type=str, default='adam')
        parser.add_argument('--s_beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--s_beta2', type=float, default=0.99, help='momentum term of adam')
        parser.add_argument('--s_lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--s_batch_size_per_res', type=int, nargs="+", default=None, help='overrides batch_size to have a different batch size for every res')
        parser.add_argument('--s_iter_function_per_res', type=str, nargs="+", default=None, help='overrides iter_function to have a different iter function for every res')
        parser.add_argument('--s_step_mul_per_res', type=float, nargs="+", default=None, help='step multiplier for every res (more epochs for specified res)')

        # for display and checkpointing
        parser.add_argument('--s_log_per_phase', type=int, default=50, help='number of times logger is updated with images during each phase, overrides log_freq')
        parser.add_argument('--s_save_at_every_res', action='store_true', help='save checkpoint when done training at a given res and moving to the next one')

        # for loading
        parser.add_argument('--s_load_path', type=str, default=None, help='load model from which_iter at specified folder')
        parser.add_argument('--s_cont_train', action='store_true', help='continue training with model from which_iter')
        parser.add_argument('--s_which_iter', type=int, default=0, help='load the model from specified iteration')
        parser.add_argument('--s_force_res', type=int, default=None, help='train model from given res (instead of estimating res from iter)')
        parser.add_argument('--s_force_phase', type=str, default=None, help='train model from given phase (instead of estimating phase from iter)')
        parser.add_argument('--s_not_strict', action='store_true', help='whether checkpoint exactly matches network architecture')

        # for output
        parser.add_argument('--s_t', type=float, default=1, help='temperature in softmax')
        parser.add_argument('--s_store_masks', action='store_true', help='to keep the masks information in the output')

        # for completor
        parser.add_argument('--s_vertical_sem_crop', action='store_true', help='if true, crop a random vertical band from sem')
        parser.add_argument('--s_min_sem_crop', type=float, default=0.5, help='min prop of image to crop for vertical sem crop')
        parser.add_argument('--s_sem_label_crop', type=int, nargs="+", default=[], help='class idx to be cropped')
        parser.add_argument('--s_sem_label_ban', type=int, nargs="+", default=[], help='class idx to be banned from the generation process')
        parser.add_argument('--s_switch_cond', action='store_true', help='if true, switch from input image cond to target cond')
        parser.add_argument('--s_fill_crop_only', action='store_true', help='if true, keep original sem and only replace cropped areas with new sem')
        parser.add_argument('--s_norm_G', type=str, default='spectralspadebatch3x3', help='instance normalization or batch normalization')
        parser.add_argument('--s_lambda_novelty', type=float, default=1., help='parameter for novelty loss')
        parser.add_argument('--s_edge_cond', action='store_true', help='if true, compute target cond by looking at edge of crop')
        parser.add_argument('--s_weight_cond_crop', action='store_true', help='if true, weight the sem cond so that it fills the crop')
        parser.add_argument('--s_bias_sem', type=int, nargs="+", default=[], help='bias some classes when filling crop')
        parser.add_argument('--s_bias_mul', type=float, default=1., help='bias mul to bias some classes when filling crop')
        parser.add_argument('--s_merged_activation', action='store_true', help='if true, merge input sem and generated sem in activation')
        parser.add_argument('--s_random_soft_mix', action='store_true', help='if true, some tgt code will be close to src')
        parser.add_argument('--s_random_linear', action='store_true', help='if true, some tgt code will be close to src')
        parser.add_argument('--s_scalnovelty', action='store_true', help='if true, novelty loss based on bhattacharyya distance')

        # for style gan 2
        parser.add_argument('--s_style_dim', type=int, default=512, help='latent dimension')
        parser.add_argument('--s_n_mlp', type=int, default=8, help='number of mlp layers')
        parser.add_argument('--s_mixing', type=float, default=0.9, help='number of mlp layers')


        return parser

    def initialize_img_generator(self, parser):
        # experiment specifics
        parser.add_argument('--i_model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--i_img_type', type=str, default='generator', help='(generator | style_generator)')
        parser.add_argument('--i_norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--i_norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--i_norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')

        # for generator
        parser.add_argument('--i_netG', type=str, default='spade', help='selects model to use for netG (condconv | pix2pixhd | spade)')
        parser.add_argument('--i_ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--i_init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--i_init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--i_latent_dim', type=int, default=256, help="dimension of the latent z vector")
        parser.add_argument('--i_num_upsampling_layers', choices=('normal', 'more', 'most'), default='normal', help="if 'more', adds upsampling layer between the two middle resnet blocks, if 'most', also add one more upsampling + resnet layer at the end of the generator")
        parser.add_argument('--i_resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--i_resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--i_resnet_kernel_size', type=int, default=3, help='kernel size of the resnet block')
        parser.add_argument('--i_resnet_initial_kernel_size', type=int, default=7, help='kernel size of the first convolution')

        # for discriminator
        parser.add_argument('--i_netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
        parser.add_argument('--i_num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        parser.add_argument('--i_n_layers_D', type=int, default=3, help='# layers in each discriminator')

        # for instance-wise features
        parser.add_argument('--i_panoptic', action='store_true', help='if true, conditioned on panoptic segmentation, semantic segmentation otherwise')
        parser.add_argument('--i_instance_type_for_img', type=str, default=None, help='combination of (center_offset | (soft_)edge | density), if None same as instance_type')
        parser.add_argument('--i_nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--i_use_vae', action='store_true', help='enable training with an image encoder.')

        # for training
        parser.add_argument('--i_optimizer', type=str, default='adam')
        parser.add_argument('--i_beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--i_beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--i_lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--i_D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        # for loading
        parser.add_argument('--i_load_path', type=str, default=None, help='load a model from specified folder')
        parser.add_argument('--i_load_path_d2', type=str, default=None, help='load a model from specified folder')
        parser.add_argument('--i_cont_train', action='store_true', help='continue training with model from which_iter')
        parser.add_argument('--i_which_iter', type=int, default=0, help='load the model from specified iteration')
        parser.add_argument('--i_which_iter_d2', type=int, default=0, help='load the model from specified iteration')
        parser.add_argument('--i_not_strict', action='store_true', help='whether checkpoint exactly matches network architecture')

        # for discriminators
        parser.add_argument('--i_ndf', type=int, default=64, help='# of discriminator filters in first conv layer')
        parser.add_argument('--i_lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--i_lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--i_no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--i_no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--i_gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--i_netD', type=str, default='multiscale', help='(fpse|n_layers|multiscale|image)')
        parser.add_argument('--i_no_TTUR', action='store_true', help='use TTUR training scheme')
        parser.add_argument('--i_lambda_kld', type=float, default=0.05)
        parser.add_argument('--i_use_d2', action='store_true', help='if true, use an additional discriminator to distinguish real and fake')
        parser.add_argument('--i_lambda_d2', type=float, default=1.0, help='weight for d2 loss')

        # for style generator
        parser.add_argument('--i_status', type=str, default='train', help='status for ACE layer')

        return parser

    def initialize_segmentor(self, parser):
        # experiment specifics
        parser.add_argument('--x_model', type=str, default='pspnet', help='(pspnet | deeplabv3)')
        parser.add_argument('--x_segment_eval_classes_only', action="store_true", help="reduce the classes for the segmentor to the eval classes")

        # training
        parser.add_argument('--x_optimizer', type=str, default='sgd')
        parser.add_argument('--x_lr', type=float, default=0.01, help='initial learning rate for adam')
        parser.add_argument('--x_momentum', type=float, default=0.9, help='momentum component of the optimiser')
        parser.add_argument("--x_not_restore_last", action="store_true", help="if specified, do not restore last (FC) layers")
        parser.add_argument("--x_power", type=float, default=0.9, help="decay parameter to compute the learning rate")
        parser.add_argument("--x_weight_decay", type=float, default=0.0005, help="regularisation parameter for L2-loss")
        parser.add_argument("--x_ohem", action="store_true", help="use hard negative mining")
        parser.add_argument("--x_ohem_thres", type=float, default=0.6, help="choose the samples with correct probability under the threshold")
        parser.add_argument("--x_ohem_keep", type=int, default=200000, help="choose the samples with correct probability under the threshold")

        # for loading
        parser.add_argument('--x_load_path', type=str, default=None, help='load a model from specified folder')
        parser.add_argument('--x_cont_train', action='store_true', help='continue training with model from which_iter')
        parser.add_argument('--x_which_iter', type=int, default=0, help='load the model from specified iteration')
        parser.add_argument('--x_pretrained_path', type=str, default=None, help='load a pretrained model from specified path')
        parser.add_argument('--x_not_strict', action='store_true', help='whether checkpoint exactly matches network architecture')

        # for loading ensemble
        parser.add_argument('--x_is_ensemble', action='store_true', help='if true, merge predictions from ensemble of two models')
        parser.add_argument('--x_load_path_2', type=str, default=None, help='load an extra model from specified folder')
        parser.add_argument('--x_which_iter_2', type=int, default=0, help='load the model from specified iteration')

        # for setting inputs
        parser.add_argument('--x_synthetic_dataset', action='store_true', help='training dataset is streaming seg/img pairs from trained generators')
        parser.add_argument('--x_semi', action='store_true', help='only img are generated')
        parser.add_argument('--x_duo', action='store_true', help='train from synthetic and real data')
        parser.add_argument('--x_duo_cond', action='store_true', help='use base and extra datasets to get cond codes')
        parser.add_argument('--x_cond_real_tgt', action='store_true', help='start from conditioning codes from real and tgt datasets')

        # for uda
        parser.add_argument('--x_advent', action='store_true', help='to train with adversarial-entropy uda')
        parser.add_argument('--x_advent_multi', action='store_true', help='if specified, discriminate at two stages')
        parser.add_argument('--x_advent_lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--x_advent_lambda_adv_final', type=float, default=0.01, help='param for adversarial loss on final seg')
        parser.add_argument('--x_advent_lambda_adv_inter', type=float, default=0.0002, help='param for adversarial loss on intermediate seg')

        # for synthetic data pre processing
        parser.add_argument('--x_sample_fixed_crop', type=int, nargs="+", default=None, help='if specified, apply a random crop of the given size')
        parser.add_argument('--x_sample_random_crop', action='store_true', help='if specified, zoom and apply a random crop while keeping original size')

        # for segmentor plus
        parser.add_argument('--x_plus', action='store_true', help='to use segmentor plus')
        parser.add_argument('--x_separable_conv', action='store_true', help='to use separable conv in segmentor plus')
        parser.add_argument('--x_output_stride', type=int, default=16, help='output stride for segmentor plus')


        return parser

    def initialize_extra_dataset(self, parser):
        # for input / output sizes
        parser.add_argument('--d_true_dim', type=int, default=1024, help='resolution of saved images')
        parser.add_argument('--d_true_ratio', type=float, default=1.0, help='ratio width/height of saved images, final width will be max_dim * aspect_ratio')
        parser.add_argument('--d_num_semantics', type=int, default=3, help='number of semantic classes including eventual unknown class')
        parser.add_argument('--d_semantic_labels', type=str, default=[], nargs="+", help='name of the semantic class for each index')
        parser.add_argument('--d_label_nc', type=int, default=None, help='new label for unknown class if there is any')

        # for setting inputs
        parser.add_argument('--d_dataroot', type=str, default='./datasets/cityscapes/')
        parser.add_argument('--d_dataset', type=str, default=None)
        parser.add_argument('--d_data_idx_type', type=str, default='both', help='(even | odd | both)')
        parser.add_argument('--d_has_tgt', action='store_true', help='if false, tgt cond overrides true cond')
        parser.add_argument('--d_estimated_cond', action='store_true', help='if true, teach a model to generate cond and sample from it')
        parser.add_argument('--d_no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--d_resize_img', type=int, nargs="+", default=None, help='if specified, resize images once they are loaded')
        parser.add_argument('--d_resize_seg', type=int, nargs="+", default=None, help='if specified, resize segmentations once they are loaded')
        parser.add_argument('--d_max_zoom', type=float, default=1., help='parameter for augmentation method consisting in zooming and cropping')
        parser.add_argument('--d_fixed_top_centered_zoom', type=float, default=None, help='if specified, crop the image to the upper center part')
        parser.add_argument('--d_max_dataset_size', type=int, default=sys.maxsize, help='maximum # of samples allowed per dataset, if the dataset directory contains more than max_dataset_size, only a subset is loaded')

        # for panoptic mode
        parser.add_argument('--d_load_panoptic', action='store_true', help='if true, loads both instance and semantic information from segmentation maps, otherwise only semantic information')
        parser.add_argument('--d_instance_type', type=str, default='center_offset', help='combination of (center_offset | (soft_)edge | density)')
        parser.add_argument('--d_things_idx', type=int, nargs="+", default=[], help='indexes corresponding to things (by opposition to stuff)')

        # for display and checkpointing
        parser.add_argument('--d_colormat', type=str, default='', help='name of colormat to display semantic maps')

        # for estimator
        parser.add_argument('--d_estimator_load_path', type=str, default=None, help='load an estimator model from specified folder')

        # for evaluation
        parser.add_argument('--d_eval_idx', type=int, nargs="+", default=[], help="selected classes for evaluation")

        return parser

    def update_defaults(self, opt, parser):
        # for base options_spade
        if opt.dim == -1:
            parser.set_defaults(dim=opt.max_dim)
        if opt.seg_dim == -1:
            seg_dim_default = opt.dim if opt.dim != -1 else opt.max_dim
            parser.set_defaults(seg_dim=seg_dim_default)
        if opt.dataset == "cityscapes":
            parser.set_defaults(dataroot="datasets/cityscapes")
            parser.set_defaults(num_semantics=35)
            parser.set_defaults(label_nc=34)
            parser.set_defaults(true_ratio=2.0)
            parser.set_defaults(i_num_upsampling_layers='more')
            parser.set_defaults(things_idx=[24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
            parser.set_defaults(semantic_labels=SEM_CITYSCAPES)
            parser.set_defaults(colormat="cityscapes_color35")
            parser.set_defaults(true_dim=1024)
            parser.set_defaults(no_h_flip=True)
            parser.set_defaults(eval_idx=[7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
        if opt.dataset == "idd":
            parser.set_defaults(dataroot="datasets/idd")
            parser.set_defaults(resize_img=[720, 1280])
            parser.set_defaults(num_semantics=40)
            parser.set_defaults(label_nc=35)
            parser.set_defaults(true_ratio=1.77777777777)
            parser.set_defaults(i_num_upsampling_layers='more')
            parser.set_defaults(things_idx=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
            parser.set_defaults(semantic_labels=SEM_IDD)
            parser.set_defaults(colormat="idd_color40")
            parser.set_defaults(true_dim=720)
            parser.set_defaults(no_h_flip=True)
            parser.set_defaults(eval_idx=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
        if opt.dataset == "celeba":
            parser.set_defaults(dataroot="datasets/celeba")
            # parser.set_defaults(dataroot="/datasets_local/CelebAMask-HQ/CelebAMask-HQ")
            parser.set_defaults(num_semantics=19)
            parser.set_defaults(label_nc=0)
            parser.set_defaults(true_ratio=1.0)
            parser.set_defaults(i_num_upsampling_layers='normal')
            parser.set_defaults(semantic_labels=SEM_CELEBA)
            parser.set_defaults(colormat="celeba_color19")
            parser.set_defaults(true_dim=512)
            parser.set_defaults(no_h_flip=True)
            parser.set_defaults(aspect_ratio=1)
            parser.set_defaults(resize_img=[512, 512])
        # for extra dataset
        if opt.d_dataset == "cityscapes":
            parser.set_defaults(d_dataroot="datasets/cityscapes")
            parser.set_defaults(d_num_semantics=35)
            parser.set_defaults(d_label_nc=34)
            parser.set_defaults(d_true_ratio=2.0)
            parser.set_defaults(d_things_idx=[24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
            parser.set_defaults(d_semantic_labels=SEM_CITYSCAPES)
            parser.set_defaults(d_colormat="cityscapes_color35")
            parser.set_defaults(d_true_dim=1024)
            parser.set_defaults(d_no_h_flip=True)
            parser.set_defaults(d_eval_idx=[7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
        if opt.d_dataset == "idd":
            parser.set_defaults(d_dataroot="datasets/idd")
            parser.set_defaults(d_resize_img=[720, 1280])
            parser.set_defaults(d_num_semantics=40)
            parser.set_defaults(d_label_nc=35)
            parser.set_defaults(d_true_ratio=1.77777777777)
            parser.set_defaults(d_things_idx=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
            parser.set_defaults(d_semantic_labels=SEM_IDD)
            parser.set_defaults(d_colormat="idd_color40")
            parser.set_defaults(d_true_dim=720)
            parser.set_defaults(d_no_h_flip=True)
            parser.set_defaults(d_eval_idx=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])

        # for img generator options_spade
        if opt.i_instance_type_for_img is None:
            parser.set_defaults(i_instance_type_for_img=opt.instance_type)
        if opt.i_netG == "spade":
            parser.set_defaults(i_norm_G='spectralspadesyncbatch3x3')
        if opt.i_netG == "condconv":
            parser.set_defaults(i_norm_G='spectralbatch')
        if opt.i_netG == "pix2pixhd":
            parser.set_defaults(i_norm_G='instance')

        return parser

    def gather_options(self):
        # initialize parser with basic options_spade
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get options_spade
        opt = parser.parse_args()

        # modify some defaults based on parser options_spade
        parser = self.update_defaults(opt, parser)
        opt = parser.parse_args()

        # if there is opt_file, load it.
        # The previous default options_spade will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)
        opt = parser.parse_args()

        self.parser = parser
        return opt

    def print_options(self, opt, opt_type, opt_prefix=""):
        def dash_pad(s, length=50):
            num_dash = max(length - len(s) // 2, 0)
            return '-' * num_dash
        opt_str = opt_type + " Options"
        message = ''
        message += dash_pad(opt_str) + ' ' + opt_str + ' ' + dash_pad(opt_str) + '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(opt_prefix + k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        end_str = opt_type + " End"
        message += dash_pad(end_str) + ' ' + end_str + ' ' + dash_pad(end_str) + '\n'
        print(message)

    def option_file_path(self, opt, signature, makedir=False):
        expr_dir = os.path.join(opt.save_path, "checkpoints", signature)
        if makedir:
            utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt, signature):
        file_name = self.option_file_path(opt, signature, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def split_options(self, opt):
        base_opt = Namespace()
        seg_generator_opt = Namespace()
        img_generator_opt = Namespace()
        segmentor_opt = Namespace()
        extra_dataset_opt = Namespace()
        for k, v in sorted(vars(opt).items()):
            if k.startswith("s_"):
                setattr(seg_generator_opt, k[2:], v)
            elif k.startswith("i_"):
                setattr(img_generator_opt, k[2:], v)
            elif k.startswith("x_"):
                setattr(segmentor_opt, k[2:], v)
            elif k.startswith("d_"):
                setattr(extra_dataset_opt, k[2:], v)
            else:
                setattr(base_opt, k, v)
        return base_opt, seg_generator_opt, img_generator_opt, segmentor_opt, extra_dataset_opt

    def copy_options(self, target_options, source_options, new_only=False):
        for k, v in sorted(vars(source_options).items()):
            if not (new_only and k in target_options):
                setattr(target_options, k, v)

    def override_num_semantics(self, opt):
        if opt.override_num_semantics is not None:
            print(f"Overriding num_semantics from {opt.num_semantics} to {opt.override_num_semantics}")
            opt.num_semantics = opt.override_num_semantics

    def set_cond_dim(self, opt):
        if opt.cond_seg == "semantic":
            cond_dim = opt.num_semantics
        elif opt.cond_seg == "instance":
            cond_dim = opt.num_things
        elif opt.cond_seg == "panoptic":
            cond_dim = opt.num_semantics + opt.num_things
        else:
            cond_dim = 0
        opt.cond_dim = cond_dim

    def set_seg_size(self, opt):
        size = opt.num_semantics
        if opt.panoptic:
            if "density" in opt.instance_type:
                size += opt.num_things
            if "center_offset" in opt.instance_type:
                size += 3
            if "edge" in opt.instance_type:
                size += 1
        opt.seg_size = size

    def parse(self, load_seg_generator=False, load_img_generator=False, load_segmentor=False,
              load_extra_dataset=False, save=False):
        opt = self.gather_options()
        signature = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "-" + opt.name

        base_opt, seg_generator_opt, img_generator_opt, segmentor_opt, extra_dataset_opt = self.split_options(opt)

        if base_opt.local_rank == 0:
            if save:
                self.save_options(opt, signature)
            self.print_options(base_opt, "Base")
            if load_seg_generator:
                self.print_options(seg_generator_opt, "Segmentation Generator", "s_")
            if load_img_generator:
                self.print_options(img_generator_opt, "Image Generator", "i_")
            if load_segmentor:
                self.print_options(segmentor_opt, "Segmentor", "x_")
            if load_extra_dataset:
                self.print_options(extra_dataset_opt, "Extra dataset", "d_")

        # set gpu ids
        str_ids = base_opt.gpu_ids.split(',')
        base_opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                base_opt.gpu_ids.append(id)

        # set num of things
        base_opt.num_things = len(base_opt.things_idx)
        extra_dataset_opt.num_things = len(extra_dataset_opt.things_idx)

        # set additional paths
        base_opt.checkpoint_path = os.path.join(base_opt.save_path, "checkpoints", signature)
        base_opt.log_path = os.path.join(base_opt.save_path, "logs", signature)

        assert (base_opt.max_dim & (base_opt.max_dim - 1)) == 0, f"Max dim {base_opt.max_dim} must be power of two."

        # set width size
        if base_opt.fixed_crop is None:
            base_opt.width_size = int(base_opt.dim * base_opt.aspect_ratio)
            base_opt.height_size = int(base_opt.width_size / base_opt.aspect_ratio)
        else:
            base_opt.height_size, base_opt.width_size = base_opt.fixed_crop

        # set semantic labels
        if len(base_opt.semantic_labels) == 0:
            base_opt.semantic_labels = ["noname"] * base_opt.num_semantics

        # set sem_conv
        if seg_generator_opt.sem_conv is not None:
            def pairwise(iterable):
                "s -> (s0, s1), (s2, s3), (s4, s5), ..."
                a = iter(iterable)
                return zip(a, a)
            seg_generator_opt.sem_conv = {i: j for i, j in pairwise(seg_generator_opt.sem_conv)}

        # set stuff idx
        base_opt.stuff_idx = [i for i in range(base_opt.num_semantics) if i not in base_opt.things_idx]

        # set signature
        base_opt.signature = signature

        self.copy_options(seg_generator_opt, base_opt)
        self.copy_options(img_generator_opt, base_opt)
        self.copy_options(segmentor_opt, base_opt)
        self.copy_options(extra_dataset_opt, base_opt, new_only=True)

        # set num semantics
        self.override_num_semantics(seg_generator_opt)

        # set cond dim
        self.set_cond_dim(seg_generator_opt)

        # set seg size
        self.set_seg_size(seg_generator_opt)

        self.base_opt = base_opt
        self.seg_generator_opt = seg_generator_opt if load_seg_generator else None
        self.img_generator_opt = img_generator_opt if load_img_generator else None
        self.segmentor_opt = segmentor_opt if load_segmentor else None
        self.extra_dataset_opt = extra_dataset_opt if load_extra_dataset else None

        self.opt = {"base": self.base_opt,
                    "seg_generator": self.seg_generator_opt,
                    "img_generator": self.img_generator_opt,
                    "segmentor": self.segmentor_opt,
                    "extra_dataset": self.extra_dataset_opt}

        return self.opt