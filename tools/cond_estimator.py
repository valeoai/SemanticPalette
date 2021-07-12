import os
import pickle
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from sklearn.mixture import GaussianMixture


class CondEstimator:
    def __init__(self, opt, max_instances=30, detection_threshold=0.005):
        self.min_components = opt.estimator_min_components
        self.max_components = opt.estimator_max_components
        self.force_components = opt.estimator_force_components
        self.projection_mode = opt.estimator_projection_mode
        self.iter_data = opt.estimator_iter_data
        self.filter_idx = opt.estimator_filter_idx
        self.n_init = opt.estimator_n_init
        l = opt.estimator_force_min_class_p
        self.force_min_class_p = [(int(l[2 * i]), l[2 * i + 1]) for i in range(len(l) // 2)]
        self.gmm = self.load_model(opt)
        self.panoptic = opt.load_panoptic
        self.max_instances = max_instances
        self.detection_threshold = detection_threshold
        self.num_semantics = opt.num_semantics
        self.opt = opt
        if self.is_fitted():
            self.o_weights = self.gmm.weights_
            self.o_means = self.gmm.means_
            self.o_covariances = self.gmm.covariances_
            size = opt.num_semantics + opt.num_things if self.panoptic else opt.num_semantics
            bias = np.zeros(size)
            bias[self.opt.estimator_force_bias] = 1
            self.update_means(bias, opt.sampler_bias_mul)

    def fit(self, source_dataloader):
        print("Fitting cond estimator to source dataset")
        sem_cond = []
        ins_cond = []

        for i in range(self.iter_data):
            for input_dict in tqdm(iter(source_dataloader), desc=f"[estimator initialization {i + 1}/{self.iter_data}] retrieving cond codes"):
                sem_cond.append(deepcopy(input_dict["sem_cond"]).numpy())
                if self.panoptic:
                    ins_cond.append(deepcopy(input_dict["ins_cond"]).numpy())

        x = np.concatenate(sem_cond, axis=0)

        if self.panoptic:
            ins_cond = np.concatenate(ins_cond, axis=0)
            x_ins = ins_cond / self.max_instances
            x = np.concatenate([x, x_ins], axis=1)

        if self.force_components is not None:
            n_components = self.force_components
            print(f"Forcing number of components for GMM to: {n_components}")
        else:
            n_components = np.arange(self.min_components, self.max_components + 1)
            models = [GaussianMixture(n, covariance_type='full').fit(x) for n in tqdm(n_components, desc="[estimator training] trying various configurations")]
            aic = [m.aic(x) for m in models]
            plt.figure()
            plt.plot(n_components, aic)
            plt.title("Akaike information criterion")
            plt.xlabel('n_components')
            save_file = os.path.join(self.opt.checkpoint_path, "aic.pdf")
            plt.savefig(save_file)
            plt.close()
            n_components = n_components[np.argmin(aic)]
            print(f"Optimal number of components for GMM is: {n_components}")

        self.gmm = GaussianMixture(n_components, covariance_type='full', n_init=self.n_init)
        self.gmm.fit(x)
        self.o_weights = self.gmm.weights_
        self.o_means = self.gmm.means_
        self.o_covariances = self.gmm.covariances_

    def sample(self, n_samples=1, return_components=False):
        x, y = self.gmm.sample(n_samples)

        if self.panoptic:
            x_sem = x[:, :self.num_semantics]
            x_ins = x[:, self.num_semantics:]
        else:
            x_sem = x
            x_ins = None

        if self.projection_mode == 'approx':
            sem_cond = self.approx_p_to_sem_cond(x_sem)
        elif self.projection_mode == 'iter':
            sem_cond = self.iter_p_to_sem_cond(x_sem)
        else:
            raise ValueError

        if x_ins is not None:
            ins_cond = (x_ins * self.max_instances).astype(int)
            thing_size = sem_cond[:, self.opt.things_idx]
            is_present = thing_size > self.detection_threshold
            ins_min = is_present.astype(int)
            ins_max = np.ones_like(is_present) * np.inf
            is_absent = np.invert(is_present)
            ins_max[is_absent] = 0
            ins_cond = np.clip(ins_cond, a_min=ins_min, a_max=ins_max)
            ins_cond = torch.tensor(ins_cond).float()

            thing_size[is_absent] = 0
            sem_cond[:, self.opt.things_idx] = thing_size
            sem_cond /= np.sum(sem_cond, axis=1, keepdims=True)
        else:
            ins_cond = torch.tensor([])

        sem_cond = torch.tensor(sem_cond).float()

        if return_components:
            return (sem_cond, ins_cond), y
        return sem_cond, ins_cond

    def approx_p_to_sem_cond(self, x):
        sem_cond = np.clip(x, a_min=0, a_max=1)
        for idx, p in self.force_min_class_p:
            sem_cond[sem_cond[:, idx] < p, idx] = p
        sem_cond[:, self.filter_idx] = 0
        sem_cond /= np.sum(sem_cond, axis=1, keepdims=True)
        return sem_cond

    def iter_p_to_sem_cond(self, x):
        n = x.shape[1]
        A = np.zeros((n + 1, n))
        A[:n] = np.eye(n)
        A[n] = 1
        lower_bound = np.zeros(n + 1)
        lower_bound[n] = 1
        upper_bound = np.ones(n + 1)
        linear_constraints = LinearConstraint(A, lower_bound, upper_bound)
        for i in range(len(x)):
            xi = x[i]
            f = lambda t: np.dot(t - xi, t - xi) / 2
            grad_f = lambda t: t - xi
            hess_f = lambda t: np.eye(n)
            res = minimize(f, xi, method='trust-constr', jac=grad_f, hess=hess_f, constraints=linear_constraints)
            x[i] = res.x
        sem_cond = np.clip(x, a_min=0, a_max=1)
        sem_cond /= np.sum(sem_cond, axis=1, keepdims=True)
        return sem_cond

    def load_model(self, opt):
        if opt.estimator_load_path is not None:
            print(f"Loading trained cond estimator model from {opt.estimator_load_path}")
            load_file = os.path.join(opt.estimator_load_path, 'cond_estimator.pkl')
            assert os.path.exists(load_file), f"File not found: {load_file}"
            with open(load_file, 'rb') as f:
                return pickle.load(f)
        else:
            return None

    def save_model(self):
        assert self.gmm is not None, "Cannot save undefined estimator"
        save_file = os.path.join(self.opt.checkpoint_path, 'cond_estimator.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(self.gmm, f)

    def plot_samples(self, n_samples):
        sem_cond, ins_cond = self.sample(n_samples)
        for i in range(len(sem_cond)):
            self.plot(sem_cond[i], "sem", i)
        if self.panoptic:
            for i in range(len(ins_cond)):
                self.plot(ins_cond[i], "ins", i)

    def plot(self, cond, mode, i):
        labels = self.opt.semantic_labels
        if mode == "ins":
            labels = [labels[i] for i in self.opt.things_idx]
        assert len(cond) == len(labels), f"Cond {len(cond)} and labels {len(labels)} should have the same size"
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(cond)), cond)
        ax.set_xticks(np.arange(0.0, len(labels), 1.0), minor=False)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major', labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        fig.tight_layout()
        save_file = os.path.join(self.opt.checkpoint_path, f"sample_{i}_{mode}.png")
        plt.savefig(save_file)

    def is_fitted(self):
        return self.gmm is not None

    def get_num_components(self):
        return self.gmm.n_components

    def get_means(self, original=False):
        return self.o_means if original else self.gmm.means_

    def get_weights(self, original=False):
        return self.o_weights if original else self.gmm.weights_

    def update_weights(self, weights):
        self.gmm.weights_ = weights

    def update_means(self, bias, bias_mul):
        mean_cond = self.o_means
        covar_cond = self.o_covariances
        std_cond = [np.diag(covar_cond[i]) for i in range(len(covar_cond))]
        std_cond = np.sqrt(np.stack(std_cond, axis=0))
        biased_cond = mean_cond + bias_mul * std_cond
        mean_cond = bias * biased_cond + (1 - bias) * mean_cond
        self.gmm.means_ = mean_cond
        # for i in range(len(covar_cond)):
        #     for k in range(self.num_semantics):
        #         std = std_cond[i, k] if std_cond[i, k] else 1
        #         norm_diff = (mean_cond[i, k] - self.o_means[i, k]) / std
        #         covar_cond[i, k, k] /= (1 + norm_diff)
        # self.gmm.covariances_ = covar_cond