import os
import logging
import time
from .model import *

MODELS = {}
MODELS["PluckerNetKnn"] = PluckerNetKnn


def load_model(name):
    all_models = MODELS
    if name not in all_models:
        logging.warning(f'Invalid model index. You put {name}. Options are:')
        for model in all_models:
            logging.warning('\t* {}'.format(model))
        return None
    return all_models[name]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


class Timer:
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.avg = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls
        return self.avg if average else self.diff
