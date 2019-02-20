SHELL := /bin/bash

docker_hub_latest:
	docker login -u truskovskyi -p transformer_chatbot
	docker build -t truskovskyi/transformer_chatbot:latest -f platform/Dockerfile .
	docker push truskovskyi/transformer_chatbot:latest

docker_hub_branch:
	docker login -u truskovskyi -p transformer_chatbot
	docker build -t truskovskyi/transformer_chatbot:$(CIRCLE_SHA1) -f platform/Dockerfile .
	docker push truskovskyi/transformer_chatbot:$(CIRCLE_SHA1)
