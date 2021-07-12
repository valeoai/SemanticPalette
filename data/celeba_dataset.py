import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class CelebaDataset(BaseDataset):

    def get_paths(self, opt, phase="train"):
        root = opt.dataroot
        assert phase == "train", "Only training data is available for this dataset"

        seg_dir = os.path.join(root, 'CelebAMask-HQ-mask')
        seg_paths_all = make_dataset(seg_dir, recursive=True)
        seg_paths = [p for p in seg_paths_all if p.endswith('.png')]

        img_dir = os.path.join(root, 'CelebA-HQ-img')
        img_paths_all = make_dataset(img_dir, recursive=True)
        img_paths = [p for p in img_paths_all if p.endswith('.jpg')]

        return seg_paths, None,  img_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1).split(".")[0]
        name2 = os.path.basename(path2).split(".")[0]
        return name1 ==  name2
