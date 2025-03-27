import torch
import torch.nn as nn
from arguments import OptimizationParams


class LearningOpacityTransform:

    def __init__(self, opacity_factor=1.0):
        self.opacity_factor = nn.Parameter(opacity_factor*torch.ones(1).float().cuda()).requires_grad_(True)

    def training_setup(self, training_args: OptimizationParams):
        l = [{'name': 'opacity_factor', 'params': self.opacity_factor, 'lr': training_args.gamma_lr}]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.opacity_factor,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        pass

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.opacity_factor,
         opt_dict) = model_args[:2]

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter