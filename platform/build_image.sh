#!/usr/bin/env bash

IMAGE_NAME='transformer_chatbot'

sudo docker build -t ${IMAGE_NAME} -f ./platform/Dockerfile .

neuro image push ${IMAGE_NAME}