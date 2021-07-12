import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urbangan_dir = os.path.dirname(current_dir)
sys.path.insert(0, urbangan_dir)

from tools.options import Options
from tools.cond_estimator import CondEstimator
from data import create_dataset, create_dataloader

class CondEstimatorTrainer:
    def __init__(self, opt):
        self.opt = opt

    def run(self):
        source_dataset = create_dataset(self.opt, load_seg=True)
        source_dataloader = create_dataloader(source_dataset, self.opt.batch_size, self.opt.num_workers, False)
        cond_estimator = CondEstimator(self.opt)
        cond_estimator.fit(source_dataloader)
        cond_estimator.save_model()
        cond_estimator.plot_samples(10)
        print('Training was successfully finished.')

if __name__ == "__main__":
    opt = Options().parse(save=True)
    CondEstimatorTrainer(opt["base"]).run()