#!/usr/bin/env bash

CPU=6
GPU=1
MEM=14G

IMAGE="truskovskyi/transformer_chatbot:b23172081a4556255521529555ccb07c95124b71"
DATA_ROOT="storage://truskovskiyk/convai"

CMD="python train.py"

non_preemptible=(experiment1 experiment2 experiment3_dual_model experiment4_single_model)

for experiment_name in "${non_preemptible[@]}"
do
    echo "run $experiment_name"
    cat ./platform/configurations/$experiment_name

    neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-v100 \
        --http 8080 --ssh 22  --no-wait-start \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/metrics/3rdparty/:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --non-preemptible \
        --description $experiment_name \
        --env-file ./platform/configurations/$experiment_name \
        ${IMAGE} \
        ${CMD}

done


preemptible=(experiment5_dual_model_w_hits experiment6_single_model_w_hits experiment7_dual_model_w_hits_dialog_embeddings)

for experiment_name in "${preemptible[@]}"
do
    echo "run $experiment_name"
    cat ./platform/configurations/$experiment_name

    neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-v100 \
        --http 8080 --ssh 22  --no-wait-start \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/metrics/3rdparty/:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --preemptible \
        --description $experiment_name \
        --env-file ./platform/configurations/$experiment_name \
        ${IMAGE} \
        ${CMD}

done
