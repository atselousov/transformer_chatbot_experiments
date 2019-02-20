#!/usr/bin/env bash
#!/usr/bin/env bash

CPU=6
GPU=1
MEM=14G

IMAGE="truskovskyi/transformer_chatbot"
DATA_ROOT="storage://truskovskiyk/convai"

CMD="python train.py"


neuro job submit \
        --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --gpu-model nvidia-tesla-v100 \
        --http 8080 --ssh 22  \
        -e FP16=True \
        --volume ${DATA_ROOT}/meteor-1.5/:/workspace/meteor-1.5:rw \
        --volume ${DATA_ROOT}/datasets/:/workspace/datasets:rw \
        --volume ${DATA_ROOT}/parameters:/workspace/parameters:rw \
        --volume ${DATA_ROOT}/runs/:/workspace/runs:rw \
        --non-preemptible \
        ${IMAGE} \
        "${CMD}"