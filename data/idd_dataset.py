import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class IddDataset(BaseDataset):

    def get_paths(self, opt, phase="train"):
        root = opt.dataroot
        phase = 'val' if phase == 'valid' else 'train'

        seg_dir = os.path.join(root, 'preprocessed', phase)
        seg_paths_all = make_dataset(seg_dir, recursive=True)
        seg_paths = [p for p in seg_paths_all if p.endswith('_instanceIds.png')]

        img_dir = os.path.join(root, 'leftImg8bit', phase)
        img_paths_all = make_dataset(img_dir, recursive=True)
        img_paths = [p for p in img_paths_all if p.endswith('_leftImg8bit.png')]

        return seg_paths, None,  img_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        return name1.split('_')[-2] ==  name2.split('_')[-2]
