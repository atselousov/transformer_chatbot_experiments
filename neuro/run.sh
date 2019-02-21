#!/usr/bin/env bash

neuro job submit -x --http 8888 --ssh 22 --memory 8G --cpu 4 --gpu 1 \
                    --gpu-model nvidia-tesla-v100 \
                    --volume storage://rauf-kurbanov/transformer_chatbot/parameters:/workspace/parameterst:ro \
                    --volume storage://rauf-kurbanov/transformer_chatbot/checkpoints:/workspace/checkpoints:ro \
                    raufkurbanov/transformer_chatbot:latest \
