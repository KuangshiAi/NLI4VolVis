# from icecream import install
# install()
# ic.configureOutput(includeContext=True) # type: ignore

import os
import torch
import torch.nn.functional as F
import torchvision
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
from scene.gamma_trans import LearningGammaTransform
from scene.opacity_trans import LearningOpacityTransform
from scene.palette_color import LearningPaletteColor
from utils.graphics_utils import hdr2ldr
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_phong=False):
    
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


    """ Training """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        # if windows is not None:
        #     windows.render()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True
                
        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                               opt=opt, is_training=True, dict_params=pbr_kwargs)
        # ic("Program exits here")
        # exit()
        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]
        
        if is_phong: 
            #* opacity loss
            points_opacity = gaussians.get_opacity[visibility_filter]
            Lalpha_regul = points_opacity.abs().mean()
            loss += 0.001*Lalpha_regul
            
        
        loss.backward()

        with torch.no_grad():
            if pipe.save_training_vis:
                save_training_vis(viewpoint_cam, gaussians, background, render_fn,
                                  pipe, opt, first_iter, iteration, pbr_kwargs)
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            # if iteration % 10 == 0:
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            training_report(tb_writer, iteration, tb_dict,
                            scene, render_fn, pipe=pipe,
                            bg_color=background, dict_params=pbr_kwargs)

            # # densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                                opt.densify_grad_normal_threshold)
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif iteration%opt.densification_interval==0:# remove not rendered points after desify iters
                gaussians.prune(1/255, scene.cameras_extent, None) 

            # Optimizer step
            gaussians.step()
            for component in pbr_kwargs.values():
                try:
                    if isinstance(component, list):
                        for c in component:
                            c.step()
                    else:
                        component.step()
                except:
                    pass

            
            # save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, is_phong=is_phong)

            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))

                for com_name, component in pbr_kwargs.items():
                    if com_name == "palette_colors": #* only save palette color into point cloud folder
                        try:
                            # ic(component[0].palette_color)
                            torch.save((component[0].capture(), iteration),
                                    os.path.join(scene.model_path, 'point_cloud', f'iteration_{iteration}', f"{com_name}_chkpnt" + ".pth"))
                            print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        except:
                            pass
                    
                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))

    
    if is_phong:
        scene.gaussians.produce_clusters(store_dict_path=scene.model_path)
        scene.gaussians.apply_clustering(codebook_dict=scene.gaussians._codebook_dict)
        scene.save(iteration, quantised=True, half_float=True)
       

    # if dataset.eval:
    #     eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)
    #* compact saving (w/o optimizer state)
    # torch.save((gaussians.compact_capture(), iteration),
    #            os.path.join(scene.model_path, "compactChkpnt" + str(iteration) + ".pth"))


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
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser) # learning parameters
    op = OptimizationParams(parser) # optimization parameters
    pp = PipelineParams(parser) # pipeline parameters

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_false', default=True, help="use gui")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'phong'], default='render')
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    
    
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)
    
    safe_state(args.quiet)

    #* this is a PyTorch function that will detect any anomaly in the computation graph, for debug purposes
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_phong = args.type in ['phong']
    training(lp.extract(args), op.extract(args), pp.extract(args), is_phong=is_phong)

    # All done
    print("\nTraining complete.")