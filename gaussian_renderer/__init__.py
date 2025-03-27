from gaussian_renderer.render import render
from gaussian_renderer.neilf import render_neilf
from gaussian_renderer.render_inverse import render_neilf_inverse


render_fn_dict = {
    "render": render,
    "phong": render_neilf,
    "inverse": render_neilf_inverse,
}