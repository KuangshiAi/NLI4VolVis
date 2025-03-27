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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <glm/glm.hpp>
#include "reduced_3dgs.h"
#include "reduced_3dgs/kmeans.h"
#include "reduced_3dgs/redundancy_score.h"


#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "cuda_rasterizer/auxiliary.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/forward.h"
using namespace torch::indexing;

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t);


std::tuple<torch::Tensor, torch::Tensor>
Reduced3DGS::intersectionTest(
    const torch::Tensor &means3D,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &neighbours_indices,
    const torch::Tensor &sphere_radius,
    const int knn)
{
    const int P = means3D.size(0);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto bool_opts = means3D.options().dtype(torch::kBool);

    torch::Tensor rotation_matrices = torch::zeros({P, 3, 3}, float_opts);
    torch::Tensor redundancy_values = torch::zeros({P, 1}, int_opts);
    torch::Tensor intersection_mask = torch::zeros({P, knn}, bool_opts);

    buildRotationMatrix(P,
                        (float4 *)(rotations.contiguous().data<float>()),
                        (glm::mat3 *)rotation_matrices.contiguous().data<float>());
    sphereEllipsoidIntersection(P,
                                (glm::vec3 *)means3D.contiguous().data<float>(),
                                (glm::vec3 *)scales.contiguous().data<float>(),
                                (glm::mat3 *)rotation_matrices.contiguous().data<float>(),
                                neighbours_indices.contiguous().data<int>(),
                                sphere_radius.contiguous().data<float>(),
                                redundancy_values.contiguous().data<int>(),
                                intersection_mask.contiguous().data<bool>(),
                                knn);

    return std::make_tuple(redundancy_values, intersection_mask);
}

torch::Tensor Reduced3DGS::calculatePixelSize(
    const torch::Tensor &w2ndc_transforms,
    const torch::Tensor &w2ndc_transforms_inverse,
    const torch::Tensor &means3D,
    const torch::Tensor &image_height,
    const torch::Tensor &image_width)
{
    const int P = means3D.size(0);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor pixel_values = torch::full({P, 1}, 10000, float_opts);

    for (int i = 0; i < w2ndc_transforms.size(0); ++i)
    {
        transformCentersNDC(
            P,
            (glm::vec3 *)means3D.contiguous().data<float>(),
            (glm::mat4 *)w2ndc_transforms.index({i}).contiguous().data<float>(),
            (glm::mat4 *)w2ndc_transforms_inverse.index({i}).contiguous().data<float>(),
            image_height.index({i}).item<int>(),
            image_width.index({i}).item<int>(),
            pixel_values.contiguous().data<float>());
    }
    return pixel_values;
}

// This function assigns for each point the minimum redundancy value out
// of all its intersecting neighbours
std::tuple<torch::Tensor>
Reduced3DGS::assignFinalRedundancyValue(
    const torch::Tensor &redundancy_values,
    const torch::Tensor &neighbours_indices,
    const torch::Tensor &intersection_mask,
    const int knn)
{
    const int P = redundancy_values.size(0);
    auto int_opts = redundancy_values.options().dtype(torch::kInt32);
    torch::Tensor minimum_redundancy_values = torch::full({P, 1}, P, int_opts);

    findMinimumRedundancyValue(P,
                               redundancy_values.contiguous().data<int>(),
                               neighbours_indices.contiguous().data<int>(),
                               intersection_mask.contiguous().data<bool>(),
                               minimum_redundancy_values.contiguous().data<int>(),
                               knn);
    return std::make_tuple(minimum_redundancy_values);
}

// Works with 256 centers 1 dimensional data only
std::tuple<torch::Tensor, torch::Tensor>
Reduced3DGS::kmeans(
	const torch::Tensor &values,
	const torch::Tensor &centers,
	const float tol,
	const int max_iterations)
{
	const int n_values = values.size(0);
	const int n_centers = centers.size(0);
	torch::Tensor ids = torch::zeros({n_values, 1}, values.options().dtype(torch::kInt32));
	torch::Tensor new_centers = torch::zeros({n_centers}, values.options().dtype(torch::kFloat32));
	torch::Tensor old_centers = torch::zeros({n_centers}, values.options().dtype(torch::kFloat32));
	new_centers = centers.clone();
	torch::Tensor center_sizes = torch::zeros({n_centers}, values.options().dtype(torch::kInt32));

	for (int i = 0; i < max_iterations; ++i)
	{
		updateIds(
			values.contiguous().data<float>(),
			ids.contiguous().data<int>(),
			new_centers.contiguous().data<float>(),
			n_values,
			n_centers);

		old_centers = new_centers.clone();
		new_centers.zero_();
		center_sizes.zero_();

		updateCenters(
			values.contiguous().data<float>(),
			ids.contiguous().data<int>(),
			new_centers.contiguous().data<float>(),
			center_sizes.contiguous().data<int>(),
			n_values,
			n_centers);

		new_centers = new_centers / center_sizes;
		new_centers.index_put_({new_centers.isnan()}, 0.f);
		float center_shift = (old_centers - new_centers).abs().sum().item<float>();
		if (center_shift < tol)
			break;
	}

	updateIds(
		values.contiguous().data<float>(),
		ids.contiguous().data<int>(),
		new_centers.contiguous().data<float>(),
		n_values,
		n_centers);

	return std::make_tuple(ids, new_centers);
}