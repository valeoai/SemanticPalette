import os
import pickle
from copy import deepcopy
from tqdm import tqdm

import numpy as np
from sklearn.neighbors import KDTree

class NearestCondIndexor:
    def __init__(self, opt, max_instances=30, detection_threshold=0.005):
        self.opt = opt
        self.panoptic = opt.load_panoptic
        self.max_instances = max_instances
        self.detection_threshold = detection_threshold
        self.tree, self.sem_means, self.ins_means = self.load_model(opt)
        self.normalize = self.opt.indexor_normalize

    def fit(self, source_dataloader):
        print("Fitting nearest cond indexor to source dataset")
        sem_cond = []
        ins_cond = []
        self.sem_means = None
        self.ins_means = None

        for input_dict in tqdm(iter(source_dataloader), desc=f"[indexor initialization] retrieving cond codes"):
            sem_cond.append(deepcopy(input_dict["sem_cond"]).numpy())
            if self.panoptic:
                ins_cond.append(deepcopy(input_dict["ins_cond"]).numpy())

        x = np.concatenate(sem_cond, axis=0)
        self.sem_means = np.ones(self.opt.num_semantics)
        if self.normalize:
            for i in range(self.opt.num_semantics):
                detection_mask = x[:,i] > self.detection_threshold
                if np.any(detection_mask):
                    self.sem_means[i] = np.mean(x[:, i][detection_mask])
        x /= self.sem_means

        if self.panoptic:
            ins_cond = np.concatenate(ins_cond, axis=0)
            x_ins = ins_cond / self.max_instances
            self.ins_means = np.ones(self.opt.num_things)
            if self.normalize:
                for i in range(self.opt.num_semantics):
                    detection_mask = ins_cond[:, i] > 0
                    if np.any(detection_mask):
                        self.ins_means[i] = np.mean(x_ins[:, i][detection_mask])
            x = np.concatenate([x, x_ins / self.ins_means], axis=1)

        self.tree = KDTree(x)

    def get_nearest_idx(self, sem_cond, ins_cond):
        sem_cond = sem_cond.view(1, -1).numpy()
        ins_cond = ins_cond.view(1, -1).numpy()
        cond = sem_cond / self.sem_means
        if self.panoptic:
            cond = np.concatenate([cond, (ins_cond / self.max_instances) / self.ins_means], axis=1)
        return self.tree.query(cond, k=1)[1][0][0]

    def load_model(self, opt):
        if opt.indexor_load_path is not None:
            print(f"Loading trained nearest cond indexor model from {opt.indexor_load_path}")
            load_file = os.path.join(opt.indexor_load_path, 'nearest_cond_indexor.pkl')
            assert os.path.exists(load_file), f"File not found: {load_file}"
            with open(load_file, 'rb') as f:
                return pickle.load(f)
        else:
            return None, None, None

    def save_model(self):
        assert self.tree is not None, "Cannot save undefined indexor"
        save_file = os.path.join(self.opt.checkpoint_path, 'nearest_cond_indexor.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump([self.tree, self.sem_means, self.ins_means], f)

    def is_fitted(self):
        return self.tree is not None