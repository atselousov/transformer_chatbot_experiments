#!/usr/bin/env bash

CPU=6
GPU=1
MEM=24G

IMAGE="truskovskyi/transformer_chatbot:10125362ba5006bc81898131d55e70e4a3793b50"
DATA_ROOT="storage://truskovskiyk/convai"

CMD="python train.py"

non_preemptible=(origin_dual_4_epochs_full_embed_meteor_risk origin_dual_10_epochs_full_embed_meteor_risk origin_single_1_epoch_full_embed origin_single_4_epoch_full_embed origin_single_4_epochs_full_embed_meteor_risk origin_single_4_epochs_full_embed_nist4_risk origin_single_7_epochs_full_embed origin_single_10_epochs_full_embed_meteor_risk origin_single_10_epochs_full_embed_nist4_risk origin_zero_shot_4_epochs origin_zero_shot_4_epochs_risk_meteor origin_zero_shot_4_epochs_risk_nist4)

for experiment_name in "${non_preemptible[@]}"
do
    echo "run $experiment_name"

    cat ./platform/configurations-batch4/$experiment_name

    neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-k80 \
        --http 8080 --ssh 22  --no-wait-start \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/metrics/3rdparty/:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --non-preemptible \
        --description $experiment_name \
        --env-file ./platform/configurations-batch4/$experiment_name \
        ${IMAGE} \
        ${CMD}
done
