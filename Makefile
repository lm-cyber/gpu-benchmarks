# Makefile for GPU Benchmarks

.PHONY: help build run-gpu run-cpu shell run-all clean logs

# Default target
help:
	@echo "Available commands:"
	@echo "  build          - Build Docker image"
	@echo "  run-gpu        - Run benchmarks on GPU"
	@echo "  run-cpu        - Run benchmarks on CPU only" 
	@echo "  run-all        - Run all benchmarks automatically"
	@echo "  shell          - Start interactive shell in container"
	@echo "  logs           - Show logs from benchmark runs"
	@echo "  clean          - Clean up containers and images"
	@echo "  test-gpu       - Test if GPU is available in container"
	@echo ""
	@echo "Examples:"
	@echo "  make build                    # Build the Docker image"
	@echo "  make run-gpu                  # Run GPU benchmarks"
	@echo "  make shell                    # Interactive development"
	@echo "  make run-specific BENCH=pytorch_benchmark SIZE=1000"

# Build Docker image
build:
	@echo "Building Docker image..."
	docker-compose build

# Run GPU benchmarks interactively
run-gpu:
	@echo "Running GPU benchmarks..."
	docker-compose run --rm gpu-benchmarks

# Run CPU benchmarks
run-cpu:
	@echo "Running CPU benchmarks..."
	docker-compose run --rm cpu-benchmarks

# Run all benchmarks automatically
run-all:
	@echo "Running all benchmarks..."
	docker-compose run --rm run-all-benchmarks

# Start interactive shell
shell:
	@echo "Starting interactive shell..."
	docker-compose run --rm shell

# Run specific benchmark (usage: make run-specific BENCH=pytorch_benchmark SIZE=1000)
run-specific:
	@if [ -z "$(BENCH)" ]; then echo "Please specify BENCH=benchmark_name"; exit 1; fi
	@echo "Running benchmark $(BENCH) with size $(SIZE)"
	docker-compose run --rm gpu-benchmarks python run.py benchmarks/$(BENCH) $(if $(SIZE),--size $(SIZE)) $(if $(DEVICE),--device $(DEVICE),--device gpu)

# Test GPU availability
test-gpu:
	@echo "Testing GPU availability..."
	docker-compose run --rm gpu-benchmarks nvidia-smi

# Show logs
logs:
	@echo "Recent logs from results directory:"
	@ls -la results/ 2>/dev/null || echo "No results directory found"

# Clean up
clean:
	@echo "Cleaning up containers and images..."
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

# Clean results only
clean-results:
	@echo "Cleaning results directory..."
	rm -rf results/
	mkdir -p results/

# Build and run (convenience target)
all: build run-all

# Quick test - run a small benchmark to verify everything works
quick-test:
	@echo "Running quick test..."
	docker-compose run --rm gpu-benchmarks python run.py benchmarks/pytorch_benchmark --size 100 --repetitions 5

# Show Docker images and containers
status:
	@echo "Docker containers:"
	@docker ps -a
	@echo ""
	@echo "Docker images:"
	@docker images | grep gpu-bench

# Development setup
dev-setup: build
	@echo "Development setup complete. Use 'make shell' to start coding." 