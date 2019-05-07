SHELL := /bin/bash
IMAGE_NAME ?= transformer_chatbot
DATA_ROOT ?= storage://rauf-kurbanov/transformer_chatbot
JOB_NAME ?= bot-dev
LOCAL_PORT ?= 1499
CODE_FOLDER ?= storage:src/transformer_chatbot_experiments
TB_FOLDER ?= storage:/board/transformer_chatbot

docker_hub_latest:
	docker login -u truskovskyi -p transformer_chatbot
	docker build -t truskovskyi/transformer_chatbot:latest -f platform/Dockerfile .
	docker push truskovskyi/transformer_chatbot:latest

docker_hub_branch:
	docker login -u truskovskyi -p transformer_chatbot
	docker build -t truskovskyi/transformer_chatbot:$(CIRCLE_SHA1) -f platform/Dockerfile .
	docker push truskovskyi/transformer_chatbot:$(CIRCLE_SHA1)

build_and_push_docker:
	docker build -t $(IMAGE_NAME) -f platform/Dockerfile .
	neuro image push $(IMAGE_NAME):latest

inference:
	neuro submit --http 8080 --ssh 22 --memory 16G --cpu 4 --gpu 1 \
                 --gpu-model nvidia-tesla-p4 \
                 --volume $(DATA_ROOT)/parameters:/workspace/parameterst:ro \
                 --volume $(DATA_ROOT)/checkpoints:/workspace/checkpoints:ro \
                 --non-preemptible \
                 $(IMAGE)

.PHONY: run
run:
	neuro submit -P --http 8888 --gpu 1 --memory 16G --cpu 4 \
				 --gpu-model nvidia-tesla-v100 \
				 --volume $(TB_FOLDER):/board/transformer_chatbot:rw \
				 --volume storage:/transformer_chatbot:/storage/transformer_chatbot:rw \
                 --volume $(CODE_FOLDER):/src/transformer_chatbot_experiments:rw \
				 --name $(JOB_NAME) \
				 image://rauf-kurbanov/$(IMAGE_NAME)

.PHONY: port_local
port_local:
	neuro port-forward $(JOB_NAME) $(LOCAL_PORT):22

.PHONY: connect
connect:
	neuro exec -t $(JOB_NAME) bash

.PHONY: clean-code
clean-code:
	neuro rm $(CODE_FOLDER)

.PHONY: clean
clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.mypy_cache' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +

.PHONY: update-code
update-code: clean
	( make clean-code; neuro cp -r -p . $(CODE_FOLDER) )

.PHONY: tb
tb:
	neuro submit --http 8888 --memory 2G --cpu 1 --gpu 0 \
	                 --volume $(TB_FOLDER):/board:ro tensorflow/tensorflow:latest \
	                 'bash -c "while true; do timeout 180 tensorboard --logdir /board --port 8888; done"'
