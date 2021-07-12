import argparse
from glob import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def main(args):
    prepare_cityscapes_25k(args.data_root)

def prepare_cityscapes_25k(data_root):
    sem_paths = glob(os.path.join(data_root, '**/*_leftImg8bit.png'), recursive=True)
    for sem_path in tqdm(sem_paths, desc=f"convert cityscapes 25k seg maps"):
        seg = np.array(Image.open(sem_path), dtype=np.uint32)
        seg = Image.fromarray(seg, mode='I')
        seg.save(sem_path.replace("_leftImg8bit.png", "_instanceIds.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    main(args)