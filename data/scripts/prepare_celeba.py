import argparse
import os
from PIL import Image
import numpy as np

def main(args):
    data_root = args.data_root
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
                  'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    raw_mask_dir = os.path.join(data_root, "CelebAMask-HQ-mask-anno")
    preprocessed_mask_dir = os.path.join(data_root, "CelebAMask-HQ-mask")
    if not os.path.exists(preprocessed_mask_dir):
        os.mkdir(preprocessed_mask_dir)
    img_num = 30000
    for k in range(img_num):
        folder_num = int(k / 2000)
        im_base = np.zeros((512, 512), dtype=np.uint32)
        for idx, label in enumerate(label_list):
            filename = os.path.join(raw_mask_dir, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
            if os.path.exists(filename):
                print(label, idx + 1)
                sem = Image.open(filename)
                sem = np.array(sem, dtype=np.uint32)
                sem = sem[:, :, 0]
                im_base[sem != 0] = (idx + 1)
        filename_save = os.path.join(preprocessed_mask_dir, str(k) + '.png')
        seg = Image.fromarray(im_base, mode='I')
        seg.save(filename_save)
        print(filename_save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    main(args)