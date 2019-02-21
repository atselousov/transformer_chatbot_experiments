#!/usr/bin/env bash

IMAGE="registry-staging.neu.ro/rauf-kurbanov/transformer_chatbot"
DATA_ROOT="storage://rauf-kurbanov/transformer_chatbot"

neuro job submit -x --http 8080 --ssh 22 --memory 16G --cpu 4 --gpu 1 \
                    --gpu-model nvidia-tesla-v100 \
                    --volume ${DATA_ROOT}/parameters:/workspace/parameterst:ro \
                    --volume ${DATA_ROOT}/checkpoints:/workspace/checkpoints:ro \
                    --non-preemptible \
                    ${IMAGE}
