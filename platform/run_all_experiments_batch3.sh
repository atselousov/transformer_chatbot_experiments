#!/usr/bin/env bash

CPU=6
GPU=1
MEM=24G

IMAGE="truskovskyi/transformer_chatbot:e02f5ed2282e1d7a9906e7c69f2787f6801696ba"
DATA_ROOT="storage://truskovskiyk/convai"

CMD="python train.py"

echo "run first 6 experiments on v100"
#v100_non_preemptible=(origin_dual_lm_loss_share_models origin_single_no_dialog_embed origin_dual_fp16 origin_dual_no_dialog_embed origin_single origin_single_zero_shot)
v100_non_preemptible=(origin_single_no_dialog_embed origin_dual_fp16 origin_dual_no_dialog_embed origin_single origin_single_zero_shot)

for experiment_name in "${v100_non_preemptible[@]}"
do
    echo "run $experiment_name"

    neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-v100 \
        --http 8080 --ssh 22  --no-wait-start \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/metrics/3rdparty/:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --non-preemptible \
        --description $experiment_name \
        --env-file ./platform/configurations-batch3/$experiment_name \
        ${IMAGE} \
        ${CMD}
done


echo "run last experiments on k80"
k80_non_preemptible=(origin_dual origin_dual_zero_shot origin_dual_suc_attn origin_dual_cls origin_single_fp16 origin_dual_lm_loss origin_dual_bs_256)

for experiment_name in "${k80_non_preemptible[@]}"
do
    echo "run $experiment_name"

    neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-k80 \
        --http 8080 --ssh 22  --no-wait-start \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/metrics/3rdparty/:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --non-preemptible \
        --description $experiment_name \
        --env-file ./platform/configurations-batch3/$experiment_name \
        ${IMAGE} \
        ${CMD}
done


# Changed after start TODO 
# platform/configurations-batch3/origin_dual_fp16              | 2 +-
# platform/configurations-batch3/origin_dual_zero_shot         | 4 ++--
# platform/configurations-batch3/origin_single                 | 2 +-
# platform/configurations-batch3/origin_single_fp16            | 4 ++--
# platform/configurations-batch3/origin_single_no_dialog_embed | 2 +-
# platform/configurations-batch3/origin_single_zero_shot       | 6 +++---