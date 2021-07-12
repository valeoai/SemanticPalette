import numpy as np
import torch

from tools.cond_estimator import CondEstimator

class CondSampler:
    def __init__(self, opt):
        self.cond_seg = opt["seg_generator"].cond_seg
        self.num_semantics = opt["seg_generator"].num_semantics
        self.weights_method = opt["base"].sampler_weights_method
        self.weights_scale = opt["base"].sampler_weights_scale
        self.bias_method = opt["base"].sampler_bias_method
        self.bias_mul = opt["base"].sampler_bias_mul
        self.method = opt["base"].sampler_method
        self.curr_components = None
        self.img_surface = opt["base"].dim * opt["base"].dim * opt["base"].aspect_ratio
        self.normalized_weights = None

        if self.cond_seg is not None:
            self.cond_estimator = CondEstimator(opt["seg_generator"])
            assert self.cond_estimator.is_fitted(), "Cond estimator should be fitted before being used in cond sampler"
            self.num_components = self.cond_estimator.get_num_components()
            self.init_weights_and_bias()
            self.init_weights_and_bias(is_tmp=True)

    def init_weights_and_bias(self, is_tmp=False):
        if is_tmp:
            self.tmp_weights = np.zeros(self.num_components)
            self.tmp_bias = np.zeros(self.num_semantics)
            self.tmp_sample_num = np.zeros(self.num_components)
            self.tmp_class_num = np.zeros(self.num_semantics)
            self.tmp_class_surface = np.zeros(self.num_semantics)
            self.tmp_ex_class_num = np.zeros(self.num_semantics)
            self.tmp_ex_class_surface = np.zeros(self.num_semantics)
        else:
            self.weights = np.zeros(self.num_components)
            self.bias = np.zeros(self.num_semantics)

    def sample_batch(self, batch_size):
        sem_cond = torch.Tensor([])
        ins_cond = torch.Tensor([])
        if self.cond_seg is not None:
            (sem_cond, ins_cond), self.curr_components = self.cond_estimator.sample(batch_size, return_components=True)
            self.tmp_ex_class_num += (sem_cond > 0).float().sum(dim=0).numpy()
            self.tmp_ex_class_surface += sem_cond.sum(dim=0).numpy() * self.img_surface
        cond = {'sem_cond': sem_cond,
                'ins_cond': ins_cond}
        return cond

    def set_batch_delta(self, delta):
        if self.cond_seg is not None:
            batch_delta = delta["batch_delta"]
            class_delta = delta["class_delta"]
            class_num = delta["class_num"]
            class_surface = delta["class_surface"]
            for i, delta in enumerate(batch_delta):
                self.tmp_weights[self.curr_components[i]] += delta.cpu().numpy()
                self.tmp_sample_num[self.curr_components[i]] += 1
            self.tmp_bias += class_delta.cpu().numpy()
            self.tmp_class_surface += class_surface.cpu().numpy()
            self.tmp_class_num += class_num.cpu().numpy()

    def update(self, logger, log, global_iteration, save=False):
        if self.cond_seg is not None:
            # update weights
            if self.weights_method is not None:
                was_updated = self.tmp_sample_num > 0
                new_weights = self.tmp_weights[was_updated] / self.tmp_sample_num[was_updated]
                if "highlight" in self.weights_method:
                    new_weights -= new_weights.min()
                if "linear" in self.weights_method:
                    self.weights[was_updated] = new_weights
                    self.normalized_weights = self.weights / np.sum(self.weights)
                elif "exponential" in self.weights_method:
                    new_weights /= new_weights.max()
                    new_weights *= self.weights_scale
                    self.weights[was_updated] = new_weights
                    self.normalized_weights = np.exp(self.weights) / np.sum(np.exp(self.weights))
                else:
                    raise ValueError
                self.cond_estimator.update_weights(self.normalized_weights)

            # update bias
            if self.bias_method is not None:
                new_bias = self.tmp_bias
                if "highlight" in self.bias_method:
                    new_bias -= new_bias.min()
                if "linear" in self.bias_method:
                    self.bias = new_bias / np.clip(self.tmp_class_surface, a_min=1, a_max=None)
                    self.bias /= self.bias.max()
                    self.cond_estimator.update_means(self.bias, self.bias_mul)
                else:
                    raise ValueError

            if log:
                self.log(logger, global_iteration, save)

            # reinit tmp weights and bias for next epoch
            self.init_weights_and_bias(True)

    def log(self, logger, global_iteration, save=False):
        if self.normalized_weights is not None:
            logger.log_component_weights("cond_sampler/component_weights", torch.tensor(self.normalized_weights), global_iteration, save)
        else:
            logger.log_component_weights("cond_sampler/component_weights", torch.tensor(self.cond_estimator.gmm.weights_), global_iteration, save)
        logger.log_class_hist("cond_sampler/normalized_bias", torch.tensor(self.bias), global_iteration, save)
        logger.log_class_hist("cond_sampler/expected_class_surface", torch.tensor(self.tmp_ex_class_surface), global_iteration, save)
        logger.log_class_hist("cond_sampler/expected_class_samples", torch.tensor(self.tmp_ex_class_num), global_iteration, save)
        logger.log_class_hist("cond_sampler/class_surface", torch.tensor(self.tmp_class_surface), global_iteration, save)
        logger.log_class_hist("cond_sampler/class_samples", torch.tensor(self.tmp_class_num), global_iteration, save)


