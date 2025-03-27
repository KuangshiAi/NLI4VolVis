#!/usr/bin/env python
"""
stylize.py
----------
A "third-stage" script for your Gaussian model that:
1) Loads the final stage-2 checkpoint.
2) Periodically stylizes the dataset's training images using InstructPix2Pix.
   (image=cond=the dataset's current images, so each stylization builds on
    the results of the previous one.)
3) Keeps training the model on these newly stylized images.
4) Saves final (and intermediate) checkpoints.

Usage Example:
--------------
    python stylize.py \
      --root_dir /path/to/dataset \
      --output_dir /path/to/stylized_outputs \
      --ckpt /path/to/stage2/final_checkpoint.pth \
      --prompt "Turn it into a magical winter wonderland" \
      --guidance_scale 9.0 \
      --image_guidance_scale 1.0 \
      --diffusion_steps 25 \
      --stylize_epochs 10000 \
      --stylize_interval 2000

Explanation:
------------
- We'll do `stylize_epochs` epochs of training.
- Every `stylize_interval` epochs, we run IP2P on the dataset images in place,
  so they become more stylized each time.
- Then we continue training on these newly stylized images.
"""

import os
import sys
import torch
import imageio
import numpy as np
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser
import torch.nn.functional as F
import torchvision
from utils.loss_utils import ssim

# ------------------------------------------------------
# Import your existing Gaussian model & dataset logic
# (Adjust paths if needed)
# ------------------------------------------------------
from scene import Scene, GaussianModel
from gaussian_renderer import render_fn_dict
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.palette_color import LearningPaletteColor
from utils.system_utils import prepare_output_and_logger
from utils.general_utils import safe_state
from utils.graphics_utils import hdr2ldr
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
from utils.image_utils import psnr, visualize_depth
from utils.system_utils import prepare_output_and_logger

# ------------------------------------------------------
# Import InstructPix2Pix (ip2p.py)
# ------------------------------------------------------
from scene.ip2p_gs import InstructPix2Pix

def stylize_training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_phong=False):
    """
    1) Load dataset & scene
    2) Load final checkpoint from stage two
    3) Initialize IP2P
    4) Training loop up to stylize_epochs:
       - if epoch % stylize_interval == 0 => stylize dataset images in place
       - train for one epoch on these images
       - periodically save checkpoint
    """

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type) # render type check whether use pbr(neilf) or not
    scene = Scene(dataset, gaussians) # by default, randomly create 100_000 points (defined in dataset_readers:readNerfSyntheticInfo:num_pts) from the scene
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
        # gaussians.load_palette_color(args.source_path+'/train')

    elif scene.loaded_iter:
        gaussians.my_load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    # exit()
    gaussians.training_setup(opt)

    pbr_kwargs = dict()
    
    #* initialize optimizable palette color
    palette_color_transforms = []
    palette_color_transform = LearningPaletteColor()
    palette_color_transform.load_palette_color(args.source_path+'/train') #* initlaize palette color with training images
    palette_color_transform.training_setup(opt)
    palette_color_transforms.append(palette_color_transform)
    pbr_kwargs["palette_colors"] = palette_color_transforms
    

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("Initializing InstructPix2Pix...")
    ip2p = InstructPix2Pix(device=device, ip2p_use_full_precision=True)
    text_emb = ip2p.pipe._encode_prompt(
        args.prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=""
    )

    """ Training """
    print(f"Stylize_epochs: {args.stylize_epochs}, Stylize_interval: {args.stylize_interval}")


    for epoch in range(1, args.stylize_epochs + 1):

        # A) Stylize the images every 'stylize_interval' epochs
        #    Here, we re-stylize them "in place," so each new stylization
        #    is applied to the current stylized version.
        if epoch % args.stylize_interval == 0 or epoch == 1:
            print(f"[Stage 3] Epoch {epoch}: stylizing dataset images with IP2P...")
            for i, viewpoint in enumerate(tqdm(train_cameras, desc="Stylizing images")):
                # current dataset image
                cur_img = viewpoint.original_image.detach().cpu()  # shape [3,H,W]
                cur_img_batch = cur_img.unsqueeze(0)  # [1,3,H,W]

                with torch.no_grad():
                    edited = ip2p.edit_image(
                        text_embeddings=text_emb,
                        image=cur_img_batch.to(device),
                        image_cond=cur_img_batch.to(device),
                        guidance_scale=args.guidance_scale,
                        image_guidance_scale=args.image_guidance_scale,
                        diffusion_steps=args.diffusion_steps,
                        lower_bound=args.lower_bound,
                        upper_bound=args.upper_bound
                    )
                stylized = edited.squeeze(0).clamp(0,1).cpu()  # shape [3,H,W]
                # Overwrite the dataset
                viewpoint.original_image = stylized.to(device)

                # Optionally save an image
                outpath = os.path.join(args.output_dir, f"epoch{epoch:04d}_view{i:03d}.png")
                imageio.imwrite(
                    outpath,
                    (stylized.permute(1,2,0).numpy()*255).astype(np.uint8)
                )

        # B) Train one epoch on the current (stylized) dataset
        print(f"[Stage 3] Epoch {epoch}: training on the stylized images...")
        # Simple example: 
        # Shuffle cameras & do forward/backward per camera (like your typical train approach).
        viewpoint_stack = train_cameras.copy()
        random_indices = torch.randperm(len(viewpoint_stack)).tolist()

        for idx in random_indices:
            vp_cam = viewpoint_stack[idx]
            render_pkg = render_fn(vp_cam, gaussians, pipe, background,
                                   opt=opt, is_training=True, dict_params=pbr_kwargs)
            loss = render_pkg["loss"]
            loss.backward()

            gaussians.step()
            # If you have PBR transforms, step them too:
            for comp in pbr_kwargs.values():
                try:
                    if isinstance(comp, list):
                        for c in comp:
                            c.step()
                    else:
                        comp.step()
                except:
                    pass

        # C) Save checkpoint periodically
        if epoch % args.checkpoint_interval == 0 or epoch == args.stylize_epochs:
            ckpt_path = os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth")
            torch.save((gaussians.capture(), epoch), ckpt_path)

    print("[Stage 3] Repeated stylize-training complete. Final model is saved.")

def training_report(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    normal = torch.clamp(
                        render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)

                    # Phong
                    diffuse_term = torch.clamp(render_pkg.get("diffuse_term", torch.zeros_like(image)), 0.0, 1.0) # base color 
                    specular_term = torch.clamp(render_pkg.get("specular_term", torch.zeros_like(image)), 0.0, 1.0) # roughness
                    ambient_term = torch.clamp(render_pkg.get("ambient_factor", torch.zeros_like(depth)), 0.0, 1.0) # metallic
                    image_pbr = render_pkg.get("phong", torch.zeros_like(image))

                    # For HDR images
                    if render_pkg["hdr"]:
                        # print("HDR detected!")
                        image = hdr2ldr(image)
                        image_pbr = hdr2ldr(image_pbr)
                        gt_image = hdr2ldr(gt_image)
                    else:
                        image = torch.clamp(image, 0.0, 1.0)
                        image_pbr = torch.clamp(image_pbr, 0.0, 1.0)
                        gt_image = torch.clamp(gt_image, 0.0, 1.0)

                    grid = torchvision.utils.make_grid(
                        torch.stack([image, image_pbr, gt_image,
                                     opacity.repeat(3, 1, 1), depth.repeat(3, 1, 1), normal,
                                     diffuse_term, specular_term, ambient_term.repeat(3, 1, 1)], dim=0), nrow=3)

                    if tb_writer and (idx < 2):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             grid[None], global_step=iteration)

                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_pbr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


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

            if is_phong:
                visualization_list.extend([
                    render_pkg["offset_color"],
                    render_pkg["shininess"].repeat(3, 1, 1),
                    render_pkg["ambient_factor"].repeat(3, 1, 1),
                    render_pkg["diffuse_term"],
                    render_pkg["specular_term"],
                    render_pkg["phong"],
                ])

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))


def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
    if gaussians.use_phong:
        os.makedirs(os.path.join(args.model_path, 'eval', 'diffuse_color'), exist_ok=True) # base color 
        os.makedirs(os.path.join(args.model_path, 'eval', 'shininess'), exist_ok=True) # roughness
        os.makedirs(os.path.join(args.model_path, 'eval', 'light_intensity'), exist_ok=True) # metallic
        os.makedirs(os.path.join(args.model_path, 'eval', 'ambient_factor'), exist_ok=True) #lights
        os.makedirs(os.path.join(args.model_path, 'eval', 'specular_factor'), exist_ok=True) #local
        os.makedirs(os.path.join(args.model_path, 'eval', 'diffuse_term'), exist_ok=True) #global
        os.makedirs(os.path.join(args.model_path, 'eval', 'specular_term'), exist_ok=True) #visibility

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            if gaussians.use_phong:
                image = results["phong"]
            else:
                image = results["render"]

            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

            save_image(image, os.path.join(args.model_path, 'eval', "render", f"{viewpoint.image_name}.png"))
            save_image(gt_image, os.path.join(args.model_path, 'eval', "gt", f"{viewpoint.image_name}.png"))
            save_image(results["normal"] * 0.5 + 0.5,
                       os.path.join(args.model_path, 'eval', "normal", f"{viewpoint.image_name}.png"))
            if gaussians.use_phong:
                save_image(results["diffuse_color"],
                           os.path.join(args.model_path, 'eval', "diffuse_color", f"{viewpoint.image_name}.png"))
                save_image(results["shininess"],
                           os.path.join(args.model_path, 'eval', "shininess", f"{viewpoint.image_name}.png"))
                save_image(results["light_intensity"],
                           os.path.join(args.model_path, 'eval', "light_intensity", f"{viewpoint.image_name}.png"))
                save_image(results["ambient_factor"],
                           os.path.join(args.model_path, 'eval', "ambient_factor", f"{viewpoint.image_name}.png"))
                save_image(results["specular_factor"],
                           os.path.join(args.model_path, 'eval', "specular_factor", f"{viewpoint.image_name}.png"))
                save_image(results["diffuse_term"],
                           os.path.join(args.model_path, 'eval', "diffuse_term", f"{viewpoint.image_name}.png"))
                save_image(results["specular_term"],
                           os.path.join(args.model_path, 'eval', "specular_term", f"{viewpoint.image_name}.png"))

    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))

if __name__ == "__main__":
    parser = ArgumentParser("Stylization training")

    # # Basic dataset/model arguments
    # parser.add_argument("--root_dir", type=str, required=True,
    #                     help="Directory of your dataset (with training images).")
    # parser.add_argument("--output_dir", type=str, required=True,
    #                     help="Where to save stylized outputs, checkpoints, logs.")
    # parser.add_argument("--ckpt", type=str, required=True,
    #                     help="Path to the final model checkpoint from stage two.")
    lp = ModelParams(parser) # learning parameters
    op = OptimizationParams(parser) # optimization parameters
    pp = PipelineParams(parser) # pipeline parameters

    # IP2P arguments
    parser.add_argument("--prompt", type=str, default="Turn it into a cartoon",
                        help="Text prompt for InstructPix2Pix.")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Text guidance scale for IP2P.")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5,
                        help="Image guidance scale for IP2P.")
    parser.add_argument("--diffusion_steps", type=int, default=20,
                        help="Number of diffusion steps for IP2P.")
    parser.add_argument("--lower_bound", type=float, default=0.02,
                        help="Lower bound of random diffusion timesteps in IP2P.")
    parser.add_argument("--upper_bound", type=float, default=0.98,
                        help="Upper bound of random diffusion timesteps in IP2P.")

    # Repeated stylization + training
    parser.add_argument("--stylize_epochs", type=int, default=10000,
                        help="Total training epochs in third stage.")
    parser.add_argument("--stylize_interval", type=int, default=2000,
                        help="Stylize the dataset images every N epochs.")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Save model checkpoint every N epochs.")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_false', default=True, help="use gui")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'phong'], default='render')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    # If you have a custom training approach or more advanced arguments, add them here
    
    
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)
    
    safe_state(args.quiet)

    #* this is a PyTorch function that will detect any anomaly in the computation graph, for debug purposes
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_phong = args.type in ['phong']

    stylize_training(lp.extract(args), op.extract(args), pp.extract(args), is_phong=is_phong)
