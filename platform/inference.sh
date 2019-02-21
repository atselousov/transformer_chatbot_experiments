#!/usr/bin/env bash

IMAGE="raufkurbanov/transformer_chatbot:latest"
DATA_ROOT="storage://rauf-kurbanov/transformer_chatbot"

neuro job submit -x --http 8888 --ssh 22 --memory 8G --cpu 4 --gpu 1 \
                    --gpu-model nvidia-tesla-v100 \
                    --volume ${DATA_ROOT}/parameters:/workspace/parameterst:ro \
                    --volume ${DATA_ROOT}/checkpoints:/workspace/checkpoints:ro \
                    --non-preemptible \
                    ${IMAGE}
