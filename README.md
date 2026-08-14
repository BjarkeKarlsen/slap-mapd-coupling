# Hierarchical Robot RL

A production-ready implementation of hierarchical multi-agent reinforcement learning 
for robot task learning using RLlib 2.x.

## Quick Start

### Installation
```bash
# Using conda
conda env create -f environment.yml
conda activate robot-rl

# Or using pip
pip install -e .
```

### Training
```bash
robot-rl train --num-iterations 500 --num-gpus 1
```

### Evaluation
```bash
robot-rl evaluate --checkpoint ./checkpoints/latest.pt --render
```

### Docker
```bash
docker build -t robot-rl .
docker run --gpus all -it robot-rl robot-rl train
```

## Documentation
- [Architecture](ARCHITECTURE.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Training Guide](docs/TRAINING.md)