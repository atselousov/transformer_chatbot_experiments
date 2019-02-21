#!/usr/bin/env bash
#!/usr/bin/env bash

CPU=6
GPU=1
MEM=14G

IMAGE="truskovskyi/transformer_chatbot:a6cb6a1957834c8c591fa0a5e566c631b5070d60"
DATA_ROOT="storage://truskovskiyk/convai"

CMD="python train.py"
DESCRIPTION="experiment #1"
CONFIGURATION="./platform/configurations/experiment1"

neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-v100 \
        --http 8080 --ssh 22  \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/meteor-1.5:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --non-preemptible \
        --description ${DESCRIPTION} \
        --env-file ${CONFIGURATION} \
        ${IMAGE} \
        "${CMD}"


#neuro job submit --non-preemptible -c 2 -m 4GB --http 8080 -v storage://truskovskiyk/convai/runs:/data/ tensorflow/tensorflow "tensorboard --logdir /data --port 8080"
#neuro model debug --localport 12789