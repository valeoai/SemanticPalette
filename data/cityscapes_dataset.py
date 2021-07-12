import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class CityscapesDataset(BaseDataset):

    def get_paths(self, opt, phase="train"):
        root = opt.dataroot
        phase = 'val' if phase == 'valid' else 'train'

        seg_dir = os.path.join(root, 'gtFine', phase)
        seg_paths_all = make_dataset(seg_dir, recursive=True)
        if opt.load_extra and phase == 'train':
            seg_paths_all = sorted(seg_paths_all, key=lambda x:os.path.basename(x))
            seg_dir = os.path.join(root, 'extraFine')
            seg_paths_all += sorted(make_dataset(seg_dir, recursive=True), key=lambda x:os.path.basename(x))
        if opt.load_minimal_info:
            seg_paths = [p for p in seg_paths_all if p.endswith('_labelIds.png')] # _gtFine_labelIds.png
        else:
            seg_paths = [p for p in seg_paths_all if p.endswith('_instanceIds.png')]

        img_dir = os.path.join(root, 'leftImg8bit', phase)
        img_paths_all = make_dataset(img_dir, recursive=True)
        if opt.load_extra and phase == 'train':
            img_paths_all = sorted(img_paths_all, key=lambda x:os.path.basename(x))
            img_dir = os.path.join(root, 'leftImg8bit', "train_extra")
            img_paths_all += sorted(make_dataset(img_dir, recursive=True), key=lambda x:os.path.basename(x))
        img_paths = [p for p in img_paths_all if p.endswith('_leftImg8bit.png')]

        return seg_paths, None,  img_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        return '_'.join(name1.split('_')[:3]) ==  '_'.join(name2.split('_')[:3])

