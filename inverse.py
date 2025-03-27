from icecream import install
install()
ic.configureOutput(includeContext=True)

import os
import torch
import torch.nn.functional as F
import torchvision
import json
import glob
from utils.system_utils import searchForMaxIteration
from collections import defaultdict
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr, visualize_depth
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from gui import GUI
from scene.derect_light_sh import DirectLightEnv
from scene.gamma_trans import LearningGammaTransform
from scene.opacity_trans import LearningOpacityTransform
from scene.palette_color import LearningPaletteColor
from scene.light_trans import LearningLightTransform
from utils.graphics_utils import hdr2ldr
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
import numpy as np
from utils.graphics_utils import focal2fov, fov2focal
from scene.cameras import Camera
from scene.dataset_readers import load_img
from pathlib import Path

from time import time_ns
import cv2

#* for GS compression
# from scene.kmeans_quantize import Quantize_kMeans


def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                   opt=opt, is_training=False, dict_params=pbr_kwargs)

            visualization_list = [
                render_pkg["render"],
                visualize_depth(render_pkg["depth"]),
                render_pkg["opacity"].repeat(3, 1, 1),
                render_pkg["normal"] * 0.5 + 0.5,
                viewpoint_cam.original_image.cuda(),
                visualize_depth(viewpoint_cam.depth.cuda()),
                viewpoint_cam.normal.cuda() * 0.5 + 0.5,
                render_pkg["pseudo_normal"] * 0.5 + 0.5,
            ]

            
            visualization_list.extend([
                render_pkg["diffuse_color"],
                render_pkg["shininess"].repeat(3, 1, 1),
                render_pkg["ambient_factor"].repeat(3, 1, 1),
                render_pkg["diffuse_term"],
                render_pkg["specular_term"],
                render_pkg["phong"],
            ])

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))





def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

def load_ckpts_paths(source_dir):
    TFs_folders = sorted(glob.glob(f"{source_dir}/TF*"))
    TFs_names = [os.path.basename(folder) for folder in TFs_folders]
  
    ckpts_transforms = {}
    for idx, TF_folder in enumerate(TFs_folders):
        one_TF_json = {'path': None, 'palette':None, 'transform': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        ckpt_dir = os.path.join(TF_folder,"neilf","point_cloud")
        max_iters = searchForMaxIteration(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "point_cloud.ply")
        palette_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "palette_colors_chkpnt.pth")
        one_TF_json['path'] = ckpt_path
        one_TF_json['palette'] = palette_path
        ckpts_transforms[TFs_names[idx]] = one_TF_json

    return ckpts_transforms

def scene_composition(scene_dict: dict, dataset: ModelParams):
    gaussians_list = []
    for scene in scene_dict:
        gaussians = GaussianModel(dataset.sh_degree, render_type="phong")
        print("Compose scene from GS path:", scene_dict[scene]["path"])
        gaussians.my_load_ply(scene_dict[scene]["path"], quantised=True, half_float=True)
        
        torch_transform = torch.tensor(scene_dict[scene]["transform"], device="cuda").reshape(4, 4)
        gaussians.set_transform(transform=torch_transform)

        gaussians_list.append(gaussians)

    gaussians_composite = GaussianModel.create_from_gaussians(gaussians_list, dataset)
    n = gaussians_composite.get_xyz.shape[0]
    print(f"Totally {n} points loaded.")

    return gaussians_composite


    

def inverse_training(args, dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams):
    first_iter = 0
    os.makedirs(args.output, exist_ok=True)
    """
    Composing Gaussians
    """
    training_cams_json = load_json_config(os.path.join(dataset.source_path, "transforms_train.json"))
    testing_cams_json = load_json_config(os.path.join(dataset.source_path, "transforms_test.json"))
    
    scene_dict = load_ckpts_paths(args.source_dir)
    gaussians = scene_composition(scene_dict, dataset)
    gaussians.freeze_attributes()
    TFs_names = list(scene_dict.keys())
    TFs_nums = len(TFs_names)

    palette_color_transforms = []
    opacity_transforms = []
    for TFs_name in TFs_names:
        palette_color_transform = LearningPaletteColor()
        palette_color_transform.create_from_ckpt(f"{scene_dict[TFs_name]['palette']}")
        palette_color_transform.training_setup(opt)
        palette_color_transforms.append(palette_color_transform)
        
        opacity_transform = LearningOpacityTransform()
        opacity_transform.training_setup(opt)
        opacity_transforms.append(opacity_transform)
    
    lighting_transform = LearningLightTransform()
    # lighting_transform.useHeadLight = False
    # lighting_transform.training_setup(opt)
    
    if args.light_type != 'Headlight':
        lighting_transform.useHeadLight = False
    # lighting_transform.training_setup(opt)
    # exit()

    pbr_kwargs = dict()
    pbr_kwargs["palette_colors"] = palette_color_transforms
    pbr_kwargs["opacity_factors"] = opacity_transforms
    pbr_kwargs["light_transform"] = lighting_transform
        
    """ Prepare render function and bg"""
    render_fn = render_fn_dict["inverse"]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    """ Inverse Training """
    training_cams_frames = training_cams_json['frames']
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    H = 800
    W = 800
    fovx = training_cams_json["camera_angle_x"]
    fovy = focal2fov(fov2focal(fovx, W), H)
    cam_kwargs = {"FoVx": fovx, "FoVy": fovy, "H": 800, "W": 800, "cx": None, "cy": None}
    
    for iteration in progress_bar:
        # Pick a random Camera
        loss = 0
        """Input data processing"""
        training_cams_frame = training_cams_frames[randint(0, len(training_cams_frames) - 1)]
        c2w = np.array(training_cams_frame['transform_matrix'], dtype=np.float32).reshape(4, 4) 
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        image_path = os.path.join(dataset.source_path, training_cams_frame["file_path"] + '.png')
        image_name = Path(image_path).stem
        image, _ = load_img(image_path)
        bg = np.array([1, 1, 1]) if dataset.white_background else np.array([0, 0, 0])
        if image.shape[-1] == 4:
            image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).cuda()
        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                            image=image,image_name=image_name, uid=0)
        
        # Render
        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(custom_cam, gaussians, pipe, background,
                            opt=opt, is_training=True, dict_params=pbr_kwargs)

        # Loss
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]    
        loss.backward()

        with torch.no_grad():
            if pipe.save_training_vis:
                save_training_vis(custom_cam, gaussians, background, render_fn,
                                  pipe, opt, first_iter, iteration, pbr_kwargs)
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            # if iteration % 10 == 0:
            progress_bar.set_postfix(pbar_dict)
            
            #* update compoenents
            for component in pbr_kwargs.values():
                try:
                    if isinstance(component, list):
                        for c in component:
                            c.step()
                    else:
                        component.step()
                except:
                    pass
            #* save compoenents
            if iteration == args.iterations:
                for com_name, component in pbr_kwargs.items():
                    if com_name in["palette_colors", "opacity_factors"]:
                        for idx,c in enumerate(component):
                            torch.save((c.capture(), iteration),
                                os.path.join(args.output,"..",f"{com_name}_{idx}_chkpnt" + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    elif com_name == "light_transform":
                        torch.save((component.capture(), iteration),
                                os.path.join(args.output,"..",f"{com_name}_chkpnt" + ".pth"))
                print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))
    eval_render(cam_kwargs, dataset, testing_cams_json, gaussians, render_fn, pipe, background, opt, pbr_kwargs)


def eval_render(cam_kwargs,dataset, testing_cams_json, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    testing_cams_frames = testing_cams_json['frames']
    for imageIdx in range(0, len(testing_cams_frames) - 1):
        training_cams_frame = testing_cams_frames[imageIdx]
        c2w = np.array(training_cams_frame['transform_matrix'], dtype=np.float32).reshape(4, 4) 
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        image_path = os.path.join(dataset.source_path, training_cams_frame["file_path"] + '.png')
        image_name = Path(image_path).stem
        image, _ = load_img(image_path)
        bg = np.array([1, 1, 1]) if dataset.white_background else np.array([0, 0, 0])
        if image.shape[-1] == 4:
            image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).cuda()
        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=cam_kwargs['FoVx'], FoVy=cam_kwargs['FoVy'], fx=None, fy=None, cx=None, cy=None,
                            image=image,image_name=image_name, uid=0)
        
        # Render
        render_pkg = render_fn(custom_cam, gaussians, pipe, background,
                            opt=opt, is_training=False, dict_params=pbr_kwargs)
        save_image(render_pkg['phong'], os.path.join(args.output, f"{image_name}.png"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser) # learning parameters
    op = OptimizationParams(parser) # optimization parameters
    pp = PipelineParams(parser) # pipeline parameters

    parser.add_argument('-so', '--source_dir', default=None, required=True, help="the source ckpts dir")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--light_type', choices=['Headlight', 'Oribital'], default='Oribital')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render', 'inverse', 'phong'], default='inverse')
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--video', action='store_true', default=False, help="If True, output video as well.")
    parser.add_argument('--output', default="./capture_trace", help="Output dir.")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    
    
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    #* this is a PyTorch function that will detect any anomaly in the computation graph, for debug purposes
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    inverse_training(args, lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
