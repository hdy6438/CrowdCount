import numpy as np
import torch

from setting import training
from tools.loading_data import loading_data
from trainer import Trainer

# ------------prepare environment------------

np.random.seed(training.seed)
torch.manual_seed(training.seed)
torch.cuda.manual_seed(training.seed)

torch.cuda.set_device(0)

# ------------Start Training------------
cc_trainer = Trainer(loading_data)
cc_trainer.train()
