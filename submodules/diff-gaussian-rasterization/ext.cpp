/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h> // 包含torch的头文件，tensor之类的
#include "rasterize_points.h"
#include "render_equation.h"
#include "reduced_3dgs.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { // Python <-> C++ 的接口
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA); // 绑定函数, 第一个参数是python中的函数名，第二个参数是C++中的函数名
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("render_equation_forward", &RenderEquationForwardCUDA);
  m.def("render_equation_forward_complex", &RenderEquationForwardCUDA_complex);
  m.def("render_equation_backward", &RenderEquationBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("sphere_ellipsoid_intersection", &Reduced3DGS::intersectionTest);
  m.def("allocate_minimum_redundancy_value", &Reduced3DGS::assignFinalRedundancyValue);
  m.def("find_minimum_projected_pixel_size", &Reduced3DGS::calculatePixelSize);
  m.def("kmeans_cuda", &Reduced3DGS::kmeans);
}