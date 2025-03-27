import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh, eval_sh_coef
from utils.loss_utils import ssim, bilateral_smooth_loss, contentrate_loss, sparsity_loss
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling
from .diff_rasterization import GaussianRasterizationSettings, GaussianRasterizer, RenderEquation, \
    RenderEquation_complex

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    palette_color_transforms = dict_params.get("palette_colors")
    opacity_transforms = dict_params.get("opacity_factors")
    light_transform = dict_params.get("light_transform")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # ic(viewpoint_camera)
    # exit()
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics
    # ic(viewpoint_camera.image_height, viewpoint_camera.image_width)
    # exit()
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color
    #* remove all grad about GSs
    offset_color = pc.get_offset_color.detach()
    diffuse_factor = pc.get_diffuse_factor.detach()
    shininess = pc.get_shininess.detach()
    ambient_factor = pc.get_ambient_factor.detach()
    specular_factor = pc.get_specular_factor.detach()
    normal = pc.get_normal.detach()
    num_GSs_TF = pc.get_num_GSs_TF
    means3D = torch.nan_to_num(means3D) #note: there are a small number of nan in the means3D (69/1744811), remove this for training
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    light_pos = light_transform.get_light_dir()
    if light_pos is None:
        incidents_dirs = viewdirs.detach().contiguous() #* now we are using headlights
    else:
        incidents_dirs = F.normalize(light_pos, dim=-1).contiguous() #* orbital light with directional lighting
     
    brdf_color, extra_results, opacity = rendering_equation_BlinnPhong_python(
        palette_color_transforms, opacity_transforms, light_transform, opacity, offset_color, diffuse_factor, shininess, ambient_factor, specular_factor, normal, viewdirs, incidents_dirs, num_GSs_TF)


    if is_training:
        # ic(brdf_color.shape, normal.shape, base_color.shape, roughness.shape, metallic.shape)
        # features = torch.cat([brdf_color, normal, base_color, roughness, metallic], dim=-1)
        features = torch.cat([brdf_color, normal, offset_color, diffuse_factor, shininess, ambient_factor, specular_factor,
                              extra_results['offset_color_norm']], dim=-1)
    else:
        # ic(brdf_color.shape,extra_results['diffuse_render'].shape, extra_results['specular_render'].shape)
        features = torch.cat([brdf_color, normal, offset_color, diffuse_factor, shininess, ambient_factor, specular_factor,
                              extra_results['diffuse_render'].mean(dim=1),
                              extra_results['specular_render'].mean(dim=1),
                              extra_results['ambient_render'].mean(dim=1)], dim=-1)
        
    # ic(shs.shape) #* this one can not be empty
    # ic(colors_precomp.shape) #* this one is OK to be empty

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
     
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    feature_dict = {}
    if is_training:
        rendered_phong, rendered_normal, rendered_offset_color, \
            rendered_diffuse_factor, rendered_shininess, \
            rendered_amibent_factor, rendered_specular_factor, \
            rendered_offset_color_norm \
            = rendered_feature.split([3, 3, 3, 1, 1, 1, 1, 1], dim=0)
        feature_dict.update({"diffuse_factor": rendered_diffuse_factor,
                             "offset_color": rendered_offset_color,
                             "shininess": rendered_shininess,
                             "ambient_factor": rendered_amibent_factor,
                             "specular_factor": rendered_specular_factor,
                             #* below are used for calculating distribution loss
                             "ambient_values": ambient_factor,
                             "offset_color_norm": rendered_offset_color_norm,
                             })
    else:
        rendered_phong, rendered_normal, rendered_offset_color, \
            rendered_diffuse_factor, rendered_shininess, \
            rendered_amibent_factor, rendered_specular_factor, \
            rendered_diffuse_term, rendered_specular_term, rendered_ambient_term \
            = rendered_feature.split([3, 3, 3, 1, 1, 1, 1, 3, 3, 3], dim=0)

        feature_dict.update({"diffuse_factor": rendered_diffuse_factor,
                             "offset_color": rendered_offset_color,
                             "shininess": rendered_shininess,
                             "ambient_factor": rendered_amibent_factor,
                             "specular_factor": rendered_specular_factor,
                             "diffuse_term": rendered_diffuse_term,
                             "specular_term": rendered_specular_term,
                             "ambient_term": rendered_ambient_term,
                             })

    phong = rendered_phong
    rendered_phong = phong + (1 - rendered_opacity) * bg_color[:, None, None]
    # rendered_normal = rendered_normal + (1 - rendered_opacity) * bg_color[:, None, None]
    # ic(gamma_transform.gamma)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_image, 
               "phong": rendered_phong,
               "normal": rendered_normal, #* normal
               "pseudo_normal": rendered_pseudo_normal, #* pseudo normal
               "surface_xyz": rendered_surface_xyz, #* surface xyz
               "opacity": rendered_opacity, #* opacity
               "depth": rendered_depth, #* depth
               "viewspace_points": screenspace_points, #* viewspace points
               "visibility_filter": radii > 0, #* visibility filter
               "radii": radii, #* radii
               "num_rendered": num_rendered, #* num rendered
               "num_contrib": num_contrib, #* num contrib
               }

    results.update(feature_dict)

    return results


def calculate_loss(viewpoint_camera, pc, results, opt):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_phong = results["phong"] #* pbr -> phong
    gt_image = viewpoint_camera.original_image.cuda()
    loss = 0.0
    #* OK
    # ic(rendered_phong.shape, gt_image.shape)
    # exit()
    
    #* OK
    if opt.lambda_phong > 0:
        Ll1_pbr = F.l1_loss(rendered_phong, gt_image)
        ssim_val_pbr = ssim(rendered_phong, gt_image)
        tb_dict["l1_pbr"] = Ll1_pbr.item()
        tb_dict["ssim_pbr"] = ssim_val_pbr.item()
        tb_dict["psnr_pbr"] = psnr(rendered_phong, gt_image).mean().item()
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
        loss = loss + opt.lambda_phong * loss_pbr
    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render_neilf_inverse(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt)
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results

def rendering_equation_BlinnPhong_python(palette_color_transforms, opacity_transforms, light_transform, opacity, offset_color, diffuse_factor, shininess, ambient_factor, specular_factor, normals, viewdirs,
                                         incidents_dirs, num_GSs_TFs):
    
    spcular_multi, diffuse_factor_multi, ambient_multi, shininess_multi,\
        specular_offset, diffuse_factor_offset, ambient_offset, shininess_offset= light_transform.get_light_transform()

    
    guassian_nums = offset_color.shape[0]
    # ic(palette_color.shape)
    offset_color = offset_color.unsqueeze(-2).contiguous()
    diffuse_factor = diffuse_factor.unsqueeze(-2).contiguous()*diffuse_factor_multi+diffuse_factor_offset # ambient white light intensity
    shininess = shininess.unsqueeze(-2).contiguous()*shininess_multi+shininess_offset # specular white light intensity
    ambient_factor = ambient_factor.unsqueeze(-2).contiguous()*ambient_multi+ambient_offset # ambient white light intensity
    specular_factor = specular_factor.unsqueeze(-2).contiguous()*spcular_multi+specular_offset # specular white light intensity
    normals = normals.unsqueeze(-2).contiguous()
    viewdirs = viewdirs.unsqueeze(-2).contiguous()
    incident_dirs = incidents_dirs.unsqueeze(-2).contiguous()
    diffuse_color = torch.zeros_like(offset_color)

    offset_color_norm = (offset_color**2).sum(dim=-1).sum(dim=-1, keepdim=True)
    
    
    #* inverse rendering
    start_GS = 0
    for i in range(len(num_GSs_TFs)):
        end_GS = num_GSs_TFs[i] + start_GS
        diffuse_segment = offset_color[start_GS:end_GS,:,:].clone().detach().requires_grad_(True)
        diffuse_segment = diffuse_segment + palette_color_transforms[i].palette_color.clamp(0,1)
        diffuse_color[start_GS:end_GS,:,:] = diffuse_segment
        # offset_color[start_GS:end_GS,:,:] = offset_color[start_GS:end_GS,:,:]+palette_color_transforms[i].palette_color.clamp(0,1)
        if opacity_transforms is not None:
            opacity_segment = opacity[start_GS:end_GS,:].clone().detach().requires_grad_(True)
            opacity_segment = opacity_segment * opacity_transforms[i].opacity_factor
            opacity[start_GS:end_GS,:] = opacity_segment
        start_GS = end_GS

    #* diffuse color 
    cos_l = (normals*incident_dirs).sum(dim=-1, keepdim=True)
    
    diffuse_intensity = diffuse_factor*torch.abs(cos_l)
    # ic(torch.isnan(offset_color).sum(), torch.isnan(ambient_factor).sum(), torch.isnan(diffuse_intensity).sum())
    diffuse_color = (ambient_factor+diffuse_intensity).repeat(1, 1, 3)*diffuse_color
    
    diffuse_term = diffuse_intensity.repeat(1, 1, 3)*diffuse_color
    ambient_term = torch.clamp(ambient_factor.repeat(1, 1, 3)*diffuse_color,0.,1.)
    
    #* specular color
    h = F.normalize(incident_dirs + viewdirs, dim=-1) # bisector h
    cos_h = (normals*h).sum(dim=-1, keepdim=True)
    specular_intensity = specular_factor*diffuse_factor*torch.where(cos_l != 0, (torch.abs(cos_h)).pow(shininess), 0.0)
    # specular_intensity = 2*(specular_factor+1.0)*torch.where(cos_l != 0, (torch.abs(cos_h)).pow(5*(shininess+1)), 0.0) # supernova
    # specular_intensity = 2*(specular_factor+1)*torch.where(cos_l != 0, (torch.abs(cos_h)).pow(5*(shininess+1)), 0.0) # fivejet

    specular_color = specular_intensity.repeat(1, 1, 3)
    

    pbr = diffuse_color.squeeze() + specular_color.squeeze() #* seems better
    pbr = torch.clamp(pbr, 0., 1.)

    extra_results = {
    "diffuse_render": diffuse_term,
    # "specular_render": specular_color,
    "specular_render": torch.clamp(specular_color, 0., 1.),
    "ambient_render": ambient_term,
    "offset_color_norm": offset_color_norm,
    }

    return pbr, extra_results, opacity