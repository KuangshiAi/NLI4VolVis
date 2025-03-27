from icecream import install
install()
ic.configureOutput(includeContext=True) # type: ignore


import json
import os
import cv2
from gaussian_renderer import render_fn_dict
import numpy as np
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.cameras import Camera
from scene.envmap import EnvLight
from utils.graphics_utils import focal2fov, fov2focal
from utils.system_utils import searchForMaxIteration
from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_renderer.neilf_composite import sample_incident_rays
from scene.palette_color import LearningPaletteColor
from scene.opacity_trans import LearningOpacityTransform
from scene.light_trans import LearningLightTransform
import glob
from time import time_ns

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

def load_ckpts_paths(source_dir, validTFs=[]):
    TFs_folders = sorted(glob.glob(f"{source_dir}/TF*"))
    TFs_names = [os.path.basename(folder) for folder in TFs_folders]
  
    ckpts_transforms = {}
    for idx, TF_folder in enumerate(TFs_folders):
        if TFs_names[idx] not in validTFs:
            continue
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


def render_points(camera, gaussians):
    intrinsic = camera.get_intrinsics()
    w2c = camera.world_view_transform.transpose(0, 1)

    xyz = gaussians.get_xyz
    color = gaussians.get_base_color
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    xyz_cam = (xyz_homo @ w2c.T)[:, :3]
    z = xyz_cam[:, 2]
    uv_homo = xyz_cam @ intrinsic.T
    uv = uv_homo[:, :2] / uv_homo[:, 2:]
    uv = uv.long()

    valid_point = torch.logical_and(torch.logical_and(uv[:, 0] >= 0, uv[:, 0] < W),
                                    torch.logical_and(uv[:, 1] >= 0, uv[:, 1] < H))
    uv = uv[valid_point]
    z = z[valid_point]
    color = color[valid_point]

    depth_buffer = torch.full_like(render_pkg['render'][0], 10000)
    rgb_buffer = torch.full_like(render_pkg['render'], bg)
    while True:
        mask = depth_buffer[uv[:, 1], uv[:, 0]] > z
        if mask.sum() == 0:
            break
        uv_mask = uv[mask]
        depth_buffer[uv_mask[:, 1], uv_mask[:, 0]] = z[mask]
        rgb_buffer[:, uv_mask[:, 1], uv_mask[:, 0]] = color[mask].transpose(-1, -2)

    return rgb_buffer


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('-vo', '--view_config', default=None, required=True, help="the config root")
    parser.add_argument('-so', '--source_dir', default=None, required=True, help="the source ckpts dir")
    parser.add_argument('-bg', "--background_color", type=float, default=None,
                        help="If set, use it as background color")
    parser.add_argument('--transform_ckpts', type=str, default=None, help="file paths of transform of check points.")
    parser.add_argument('--video', action='store_true', default=False, help="If True, output video as well.")
    parser.add_argument('--output', default="./capture_trace", help="Output dir.")
    parser.add_argument('--capture_list',
                        default="phong,normal,diffuse_term,specular_term,ambient_term",
                        help="what should be rendered for output.")
    parser.add_argument('--useHeadlight', action='store_true', help="If True, use head light.")
    parser.add_argument('--validTFs', default="", help="validTFs for composing")
    parser.add_argument('--evaluation', action='store_false', help="If True, eval mode.")
    parser.add_argument('--EvalTime', action='store_true', help="If True, eval time, not save images.")
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    

    # load configs
    view_config_file = f"{args.view_config}/transforms_test.json"
    allTFDirs = sorted(glob.glob(f"{args.source_dir}/TF*"))
    allTFsNames = [os.path.basename(TFDir) for TFDir in allTFDirs]

    validTFs = args.validTFs.split(",") if args.validTFs else allTFsNames

    scene_dict = load_ckpts_paths(args.source_dir, validTFs=validTFs)
    TFs_names = list(scene_dict.keys())
    TFs_nums = len(TFs_names)

    view_dict = load_json_config(view_config_file)
    
    if not args.transform_ckpts:
        # load palette
        palette_color_transforms = []
        opacity_transforms = []
        for TFs_name in TFs_names:
            palette_color_transform = LearningPaletteColor()
            palette_color_transform.create_from_ckpt(f"{scene_dict[TFs_name]['palette']}")
            palette_color_transforms.append(palette_color_transform)
            # ic(palette_color_transform.palette_color)
            
            opacity_transform = LearningOpacityTransform()
            opacity_transforms.append(opacity_transform)
            
        light_transform = LearningLightTransform(theta=180, phi=0)
    else:
        palette_color_transforms = []
        opacity_transforms = []
        for idx, TFs_name in enumerate(TFs_names):
            palette_color_transform = LearningPaletteColor()
            palette_color_transform.create_from_ckpt(f"{args.transform_ckpts}/palette_colors_{idx}_chkpnt.pth")
            palette_color_transforms.append(palette_color_transform)
            # ic(palette_color_transform.palette_color)
            opacity_transform = LearningOpacityTransform()
            opacity_transform.create_from_ckpt(f"{args.transform_ckpts}/opacity_factors_{idx}_chkpnt.pth")
            opacity_transforms.append(opacity_transform)
            ic(opacity_transform.opacity_factor)
            
            
        light_transform = LearningLightTransform(theta=180, phi=0)
        light_transform.create_from_ckpt(f"{args.transform_ckpts}/light_transform_chkpnt.pth")
    # load gaussians
    gaussians_composite = scene_composition(scene_dict, dataset)

    # rendering
    capture_dir = args.output
    os.makedirs(capture_dir, exist_ok=True)
    capture_list = [str.strip() for str in args.capture_list.split(",")]
    for capture_type in capture_list:
        capture_type_dir = os.path.join(capture_dir, capture_type)
        os.makedirs(capture_type_dir, exist_ok=True)
    

    bg = args.background_color
    if bg is None:
        bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict["inverse"]

    if not args.transform_ckpts:
        render_kwargs = {
            "pc": gaussians_composite,
            "pipe": pipe,
            "bg_color": background,
            "is_training": False,
            "dict_params": {
                "sample_num": args.sample_num,
                "palette_colors": palette_color_transforms,
                "light_transform": light_transform
            }
        }
    else:
        render_kwargs = {
            "pc": gaussians_composite,
            "pipe": pipe,
            "bg_color": background,
            "is_training": False,
            "dict_params": {
                "sample_num": args.sample_num,
                "palette_colors": palette_color_transforms,
                "light_transform": light_transform,
                "opacity_factors": opacity_transforms,
            }
        }
    
    if not args.useHeadlight:
        render_kwargs["dict_params"]["light_transform"].useHeadLight = False

    H = 800
    W = 800
    fovx = 30 * np.pi / 180
    fovy = focal2fov(fov2focal(fovx, W), H)

    progress_bar = tqdm(view_dict["frames"], desc="Rendering")
    time_log = []
    for idx, cam_info in enumerate(progress_bar):
        tic = time_ns()
        # cam_info = traject_dict["trajectory"].items()
        # ic(cam_info)
        cam_pose = cam_info['transform_matrix']
        light_angle = cam_info.get('light_angle', None)
        # exit()

        c2w = np.array(cam_pose, dtype=np.float32).reshape(4, 4) 
        
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                            image=torch.zeros(3, H, W), image_name=None, uid=0)
      
        # ic(capture_list)
        # exit()
        if light_angle is not None:
            render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(light_angle[0], light_angle[1])

        with torch.no_grad():
            render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)
        if not args.EvalTime:
            for capture_type in capture_list:
                if capture_type == "points":
                    render_pkg[capture_type] = render_points(custom_cam, gaussians_composite)
                elif capture_type == "normal":
                    render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
                elif capture_type in ["base_color", "roughness", "metallic", "visibility"]:
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
                elif capture_type in ["ambient_term", "specular_term", "diffuse_term"]:
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
            
                save_image(render_pkg[capture_type], f"{capture_dir}/{capture_type}/frame_{int(idx):04d}.png")
        toc = time_ns()
        time_log.append((toc - tic)/1e9)
    time_log = np.array(time_log)
    print(f"Averaged Rendering time: {time_log.mean()*1e3} ms")
    print(f"FPS: {1/time_log.mean()}")
    
    if (args.evaluation and not args.EvalTime):
        GTImgPaths = sorted(glob.glob(f"{args.view_config}/test/*.png"))
        evalImgPaths = sorted(glob.glob(f"{capture_dir}/phong/*.png"))
        psnr_test = 0
        
        for idx, evalImgPath in enumerate(evalImgPaths):
            evalImg = cv2.imread(evalImgPath)
            GTImg = cv2.imread(GTImgPaths[idx])
            # print(f"evalImgPath: {evalImgPath}", "GTImgPath: ", GTImgPaths[idx])
            psnr_test += cv2.PSNR(GTImg, evalImg)
            # print(f"PSNR for {idx}: {cv2.PSNR(GTImg, evalImg)}")
        psnr_test /= len(evalImgPaths)
        print(f"PSNR: {psnr_test}")


    # output as video
    if (args.video and not args.EvalTime):
        progress_bar = tqdm(capture_list, desc="Outputting video")
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        for capture_type in progress_bar:
            video_path = f"{capture_dir}/{capture_type}.mp4"
            image_names = [os.path.join(capture_dir, capture_type, f"frame_{int(j):04d}.png") for j in
                           range(len(view_dict["frames"]))]
            media_writer = cv2.VideoWriter(video_path, fourcc, 60, (W, H))

            for image_name in image_names:
                img = cv2.imread(image_name)
                media_writer.write(img)
            media_writer.release()