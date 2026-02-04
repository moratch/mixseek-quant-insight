# quant-insight Minimal Environment Makefile

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_NAME := mixseek-quant-insight
IMAGE_NAME := $(APP_NAME)/minimal:latest
CONTAINER_NAME := $(APP_NAME)-minimal
ROOTDIR := $(shell cd "$(CURDIR)/../.." && pwd -P)

MIXSEEK_USERNAME ?= quant_insight
MIXSEEK_UID ?= $(shell id -u)
MIXSEEK_GID ?= $(shell id -g)

BUILD_ARGS := \
    --build-arg MIXSEEK_USERNAME=$(MIXSEEK_USERNAME) \
    --build-arg MIXSEEK_UID=$(MIXSEEK_UID) \
    --build-arg MIXSEEK_GID=$(MIXSEEK_GID)

DOCKER_OPTS := \
    -v $(ROOTDIR):/app \
    --env-file=$(CURDIR)/.env

# =============================================================================
# BUILD & RUN
# =============================================================================

.PHONY: build
build:
	cd $(ROOTDIR) && docker build -t $(IMAGE_NAME) -f $(CURDIR)/Dockerfile $(BUILD_ARGS) .

.PHONY: run
run:
	@test -f "$(CURDIR)/.env" || (echo "Error: .env not found. Run: cp .env.template .env" && exit 1)
	@if [ -n "$(ARG)" ]; then \
		docker run --rm $(DOCKER_OPTS) $(IMAGE_NAME) $(ARG); \
	else \
		docker run --name $(CONTAINER_NAME) $(DOCKER_OPTS) -itd $(IMAGE_NAME); \
	fi

.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) 2>/dev/null || true

.PHONY: bash
bash:
	docker exec -it $(CONTAINER_NAME) bash

# =============================================================================
# CLEANUP
# =============================================================================

.PHONY: rm
rm:
	docker rm -f $(CONTAINER_NAME) 2>/dev/null || true

.PHONY: rmi
rmi:
	docker rmi $(IMAGE_NAME) 2>/dev/null || true

.PHONY: clean
clean: rm rmi

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make build              Build image"
	@echo "  make run                Start container (background)"
	@echo "  make run ARG='command'  Run command and exit"
	@echo "  make stop               Stop container"
	@echo "  make bash               Connect to shell"
	@echo "  make rm                 Remove container"
	@echo "  make rmi                Remove image"
	@echo "  make clean              Remove all"
