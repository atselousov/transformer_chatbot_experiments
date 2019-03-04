#!/usr/bin/env bash

CPU=6
GPU=1
MEM=24G

IMAGE="truskovskyi/transformer_chatbot:d0d690d2f150c1281abf43f66831e1a2a8dbbdb9"
# build https://circleci.com/gh/atselousov/transformer_chatbot_experiments/190?utm_campaign=vcs-integration-link&utm_medium=referral&utm_source=github-build-link
DATA_ROOT="storage://truskovskiyk/convai"

CMD="python train.py"

non_preemptible_v100=(origin_single_1_epochs_full_embed origin_single_4_epochs_full_embed origin_single_4_epochs_high_s2s_coef origin_single_4_epochs_no_dialog_embed)

for experiment_name in "${non_preemptible_v100[@]}"
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
        --env-file ./platform/configurations-batch6/$experiment_name \
        ${IMAGE} \
        ${CMD}
done

non_preemptible_k80=(origin_single_4_epochs_no_start_end origin_single_10_epochs_full_embed origin_zero_shot_4_epochs origin_zero_shot_10_epochs)

for experiment_name in "${non_preemptible_k80[@]}"
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
        --env-file ./platform/configurations-batch6/$experiment_name \
        ${IMAGE} \
        ${CMD}
done

