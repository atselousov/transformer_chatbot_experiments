#!/usr/bin/env bash

CPU=6
GPU=1
MEM=24G

IMAGE="truskovskyi/transformer_chatbot:70580637c0ae988d49b0d5b5f28b4846796e42bb"
DATA_ROOT="storage://truskovskiyk/convai"

CMD="python train.py"

non_preemptible=(origin_single_cls origin_single_no_dialog_embed origin_dual_no_dialog_embed origin_single origin_dual_super_shared origin_single_aug origin_single_norm_embed origin_dual origin_dual_share_models origin_dual_suc_attn origin_dual_share_attn origin_dual_no_norm_embed origin_dual_super_shared_aug origin_dual_const_pos origin_dual_no_sparse_embed revised_dual_super_shared revised_single_cls)

for experiment_name in "${non_preemptible[@]}"
do
    echo "run $experiment_name"

    cat ./platform/configurations-batch2/$experiment_name

    neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-v80 \
        --http 8080 --ssh 22  --no-wait-start \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/metrics/3rdparty/:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --non-preemptible \
        --description $experiment_name \
        --env-file ./platform/configurations-batch2/$experiment_name \
        ${IMAGE} \
        ${CMD}
done
