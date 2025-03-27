import torch
import torch.nn as nn
from arguments import OptimizationParams
from tqdm import tqdm
import numpy as np

class LearningPaletteColor:

    def __init__(self, palette_color=None):
        if palette_color is not None:
            self.palette_color = nn.Parameter(palette_color.float().cuda()).requires_grad_(True)
        else:
            self.palette_color = nn.Parameter(torch.ones(3).float().cuda()).requires_grad_(True)

    def training_setup(self, training_args: OptimizationParams):
        l = [{'name': 'palette_color', 'params': self.palette_color, 'lr': training_args.palette_color_lr }]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.palette_color,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        pass

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.palette_color,
         opt_dict) = model_args[:2]

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

    def load_palette_color(self, img_dir_path):
        def extract_palette(img_dir):
            import glob
            import imageio.v2 as imageio
            img_files = sorted(glob.glob(f"{img_dir}/*.png"))
            avg_colors = []
            print(f"Computing mean color of training images as palette color with {img_dir}")
            for img_file in tqdm(img_files,leave=False):
                img = imageio.imread(img_file)
                mask = img[:, :, 3]
                colors = img[:, :, :3]  
                avg_colors.append(np.mean(colors[mask != 0], axis=0))
            avg_colors = np.array(avg_colors)
            palette = avg_colors.mean(axis=0) / 255
            return palette
            
        palette = extract_palette(img_dir_path)
        palette = torch.tensor(palette).float().to(self.palette_color.device)
        with torch.no_grad():
            self.palette_color.data = palette