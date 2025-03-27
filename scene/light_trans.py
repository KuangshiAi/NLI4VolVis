import torch
import torch.nn as nn
from arguments import OptimizationParams
import math

class LearningLightTransform:

    def __init__(self, theta=0, phi=0, useHeadLight=True):
        self.theta = theta
        self.phi = phi
        self.specular_multi = 1
        self.specular_offset = 0
        self.light_intensity_multi = 1
        self.light_intensity_offset = 0
        self.ambient_multi = 1
        self.ambient_offset = 0
        self.shininess_multi = 1
        self.shininess_offset = 0
        self.useHeadLight = useHeadLight
    
    def set_params(self):
        with torch.no_grad():
            self.theta = nn.Parameter(torch.ones(1).float().cuda()).requires_grad_(True)
            self.phi = nn.Parameter(torch.ones(1).float().cuda()).requires_grad_(True)
            self.specular_multi = nn.Parameter(torch.ones(1).float().cuda()).requires_grad_(True)
            self.specular_offset = nn.Parameter(torch.zeros(1).float().cuda()).requires_grad_(True)
            self.light_intensity_multi = nn.Parameter(torch.ones(1).float().cuda()).requires_grad_(True)
            self.light_intensity_offset = nn.Parameter(torch.zeros(1).float().cuda()).requires_grad_(True)
            self.ambient_multi = nn.Parameter(torch.ones(1).float().cuda()).requires_grad_(True)
            self.ambient_offset = nn.Parameter(torch.zeros(1).float().cuda()).requires_grad_(True)
            self.shininess_multi = nn.Parameter(torch.ones(1).float().cuda()).requires_grad_(True)
            self.shininess_offset = nn.Parameter(torch.zeros(1).float().cuda()).requires_grad_(True)
            
    def training_setup(self, training_args: OptimizationParams):
        self.set_params()
        # ic(self.theta, self.phi, self.specular_multi, self.light_intensity_multi, self.ambient_multi, self.shininess_multi)
        l = [{'name': 'theta', 'params': self.theta, 'lr': training_args.theta_lr},
             {'name': 'phi', 'params': self.phi, 'lr': training_args.phi_lr},
             {'name': 'spcular_multi', 'params': self.specular_multi, 'lr': training_args.specular_multi_lr},
             {'name': 'specular_offset', 'params': self.specular_offset, 'lr': training_args.specular_multi_lr},
             {'name': 'light_intensity_multi', 'params': self.light_intensity_multi, 'lr': training_args.light_intensity_multi_lr},
             {'name': 'light_intensity_offset', 'params': self.light_intensity_offset, 'lr': training_args.light_intensity_multi_lr},
             {'name': 'ambient_multi', 'params': self.ambient_multi, 'lr': training_args.ambient_multi_lr},
             {'name': 'ambient_offset', 'params': self.ambient_offset, 'lr': training_args.ambient_multi_lr},
             {'name': 'shininess_multi', 'params': self.shininess_multi, 'lr': training_args.shininess_multi_lr},
             {'name': 'shininess_offset', 'params': self.shininess_offset, 'lr': training_args.shininess_multi_lr},]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def get_light_transform(self):
        return self.specular_multi, self.light_intensity_multi, self.ambient_multi, self.shininess_multi, \
                self.specular_offset, self.light_intensity_offset, self.ambient_offset, self.shininess_offset
    
    def set_light_theta_phi(self, theta, phi):
        if torch.is_tensor(self.theta):
            with torch.no_grad():
                # self.theta = nn.Parameter((theta + 180) / 180 * math.pi).requires_grad_(True)
                # self.phi = nn.Parameter(phi / 180 * math.pi).requires_grad_(True)
                self.theta = theta
                self.phi = phi
        else:
            self.theta = theta
            self.phi = phi

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.theta,
            self.phi,
            self.specular_multi,
            self.light_intensity_multi,
            self.ambient_multi,
            self.shininess_multi,
            # self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        pass
    
    def get_light_dir(self):
        if self.useHeadLight:
            return None
        if torch.is_tensor(self.theta):
            x = torch.cos(self.theta)*torch.cos(self.phi)
            y = torch.sin(self.theta)*torch.cos(self.phi)
            z = torch.sin(self.phi)
            return torch.tensor([x, y, z], dtype=torch.float32, device="cuda")
        else:
            # ic("We are here")
            # exit()
            theta_rad = (self.theta + 180) / 180 * math.pi
            phi_rad = self.phi / 180 * math.pi
            x = math.cos(theta_rad) * math.cos(phi_rad)
            y = math.sin(theta_rad) * math.cos(phi_rad)
            z = math.sin(phi_rad)
            return 4.0311*torch.tensor([-x, z, y], dtype=torch.float32, device="cuda")
            
    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.theta,
         self.phi,
         self.specular_multi,
         self.light_intensity_multi,
         self.ambient_multi,
         self.shininess_multi) = model_args[:6]

        # if restore_optimizer:
        #     try:
        #         self.optimizer.load_state_dict(opt_dict)
        #     except:
        #         print("Not loading optimizer state_dict!")

        return first_iter