#!/usr/bin/env bash

CPU=6
GPU=1
MEM=24G

IMAGE="truskovskyi/transformer_chatbot:10125362ba5006bc81898131d55e70e4a3793b50"
DATA_ROOT="storage://truskovskiyk/convai"

CMD="python train.py"

non_preemptible=(origin_dual_20_epochs_no_dialog_embed origin_dual_20_epochs_zero_shot origin_single_1_epochs_full_embed origin_single_1_epochs_no_dialog_embed origin_single_1_epochs_no_start_end origin_single_4_epochs_full_embed origin_single_4_epochs_no_dialog_embed origin_single_4_epochs_no_start_end origin_single_7_epochs_full_embed origin_single_7_epochs_no_dialog_embed origin_single_7_epochs_no_start_end origin_single_10_epochs_full_embed origin_single_10_epochs_no_dialog_embed origin_single_10_epochs_no_start_end origin_zero_shot_1_epochs origin_zero_shot_4_epochs origin_zero_shot_7_epochs origin_zero_shot_10_epochs)

for experiment_name in "${non_preemptible[@]}"
do
    echo "run $experiment_name"

    cat ./platform/configurations-batch5/$experiment_name

    neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-k80 \
        --http 8080 --ssh 22  --no-wait-start \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/metrics/3rdparty/:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --non-preemptible \
        --description $experiment_name \
        --env-file ./platform/configurations-batch5/$experiment_name \
        ${IMAGE} \
        ${CMD}
done
