#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables

root_base_dir="./ImgData"

datasets=(
    "carp_boneRGBa_sags_class7"
    "supernovaRGBa_tf_class3"
    "chameleonRGBa_tf_class2"
    "mantleRGBa_tf_class5"
)

cd ..

for exp_name in "${datasets[@]}"; do
    root_dir="$root_base_dir/${exp_name}"
    echo "Dataset root dir: ${root_dir}"

    list=$(basename -a $root_dir/*/ | grep '^TF')
    echo "TFs list for ${exp_name}: $list"

    for i in $list; do
        echo "Processing: ${exp_name} -> ${i}"
        tic=$(date +%s)

        python train.py -s ${root_dir}/${i} \
            -m output/${exp_name}/${i}/3dgs \
            --lambda_normal_render_depth 0.01 \
            --lambda_opacity 0.1 \
            --densification_interval 500 \
            --densify_grad_normal_threshold 0.000004 \
            --save_training_vis

        python train.py -s ${root_dir}/${i} \
            -m output/${exp_name}/${i}/neilf \
            -c output/${exp_name}/${i}/3dgs/chkpnt30000.pth \
            -t phong \
            --lambda_normal_render_depth 0.01 \
            --lambda_opacity 0.1 \
            --lambda_phong 1.0 \
            --densify_until_iter 32000 \
            --lambda_render 0.0 \
            --use_global_shs \
            --finetune_visibility \
            --iterations 40000 \
            --test_interval 1000 \
            --checkpoint_interval 2500 \
            --lambda_offset_color_sparsity 0.01 \
            --lambda_ambient_factor_smooth 0.01 \
            --lambda_specular_factor_smooth 0.01 \
            --lambda_normal_smooth 0.00 \
            --lambda_diffuse_factor_smooth 0.01 \
            --save_training_vis

        toc=$(date +%s)
        echo "Processing ${exp_name}/${i} took $((toc - tic)) seconds" >> output/${exp_name}/${i}/time.txt
    done

done
