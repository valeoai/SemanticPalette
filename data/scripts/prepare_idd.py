import argparse
from glob import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def main(args):
    prepare_idd_for_phase(args.data_root, "train")
    prepare_idd_for_phase(args.data_root, "val")

def prepare_idd_for_phase(data_root, phase):
    sem_paths = glob(os.path.join(data_root, "gtFine", phase, '**/*_labelids.png'), recursive=True)
    for sem_path in tqdm(sem_paths, desc=f"prepare idd for {phase}"):
        fold_id = sem_path.split('/')[-2]
        sem_id = sem_path.split('_')[-3].split('/')[-1]
        ins_path = os.path.join(data_root, "gtFine", f"{phase}_panoptic", f"{fold_id}_{sem_id}_gtFine_panopticlevel3Ids.png")
        sem = Image.open(sem_path)
        seg = np.array(sem.resize((1280, 720), resample=Image.NEAREST), dtype=np.uint32)
        ins = np.array(Image.open(ins_path), dtype=np.uint32)
        r, g, b = ins[:, :, 0], ins[:, :, 1], ins[:, :, 2]
        ins = r + 256 * g + 256 * 256 * b
        new_ins = np.zeros_like(ins)
        values = np.unique(ins)
        values = [v for v in values if v != 0]
        for i, v in enumerate(values):
            new_ins[ins == v] = i
        ins = new_ins
        has_ins = np.bitwise_and(seg >= 6, seg <= 18)
        seg[has_ins] = seg[has_ins] * 1000 + ins[has_ins]
        seg = Image.fromarray(seg, mode='I')
        seg_dir = os.path.join(data_root, "preprocessed", phase, fold_id)
        os.makedirs(seg_dir, exist_ok=True)
        seg_path = os.path.join(seg_dir, f"{sem_id}_instanceIds.png")
        seg.save(seg_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    main(args)