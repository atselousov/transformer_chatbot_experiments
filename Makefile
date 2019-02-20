SHELL := /bin/bash
USER_NAME ?= truskovskiyk
USER_PASS ?= $(CI_KYRYL_TOKEN)
IMAGE_NAME ?= transformer_chatbot
IMAGE_TAG ?= latest
DOCKER_REGISTRY ?= registry.staging.neuromation.io


build_and_push_docker:
	docker login -u $(USER_NAME) -p $(USER_PASS) $(DOCKER_REGISTRY)
	docker build -t $(DOCKER_REGISTRY)/$(USER_NAME)/$(IMAGE_NAME):$(IMAGE_TAG) -f platform/Dockerfile .
	docker tag $(DOCKER_REGISTRY)/$(USER_NAME)/$(IMAGE_NAME):$(IMAGE_TAG) $(DOCKER_REGISTRY)/$(USER_NAME)/$(IMAGE_NAME):$(CIRCLE_SHA1)
	docker push $(DOCKER_REGISTRY)/$(USER_NAME)/$(IMAGE_NAME):$(CIRCLE_SHA1)

build_and_push_docker_latest:
	docker login -u $(USER_NAME) -p $(USER_PASS) $(DOCKER_REGISTRY)
	docker build -t $(DOCKER_REGISTRY)/$(USER_NAME)/$(IMAGE_NAME):$(IMAGE_TAG) -f platform/Dockerfile .
	docker push $(DOCKER_REGISTRY)/$(USER_NAME)/$(IMAGE_NAME):$(IMAGE_TAG)
