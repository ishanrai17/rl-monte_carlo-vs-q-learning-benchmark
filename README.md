# Cliff Walking RL Benchmarks

This project provides implementations of classic Reinforcement Learning (RL) algorithms—Monte Carlo and Q-Learning—on the OpenAI Gymnasium "CliffWalking-v0" environment. It is designed for benchmarking and educational purposes.

## Features

- **CliffWalking Environment**: Uses OpenAI Gymnasium's gridworld environment.
- **Algorithms**: Implements both Monte Carlo and Q-Learning methods.
- **Configurable Parameters**: Easily adjust learning rate, discount factor, epsilon, and decay rate.
- **Training & Testing**: Simple interface for training and visualizing agent performance.

## Project Structure

```
.
├── src/
│   ├── train.py
│   ├── algorithms/
│   │   ├── abstract_rl_algorithm.py
│   │   ├── constants.py
│   │   ├── monte_carlo.py
│   │   └── q_learning.py
│   └── env_builders/
│       ├── cliff_walking.py
│       └── constants.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd cliff-walking-benchmarks
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python3 -m venv open-ai-gym-cliff-walking
   source open-ai-gym-cliff-walking/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Navigate into the `src` folder:

```sh
cd src
```

Run the training and testing script:

```sh
python train.py
```

## Algorithms

1. **Monte Carlo**: Implements the Monte Carlo method for estimating the value of states and actions based on sampled returns.

### Visualization
![Monte Carlo Visualization](assets/Monte%20Carlo%2030000%20rounds.gif)

2. **Q-Learning**: Implements the Q-Learning algorithm, a model-free reinforcement learning algorithm that learns the value of actions in states.

### Visualization
![Cliff Walking Visualization](assets/Q%20learning%203000%20rounds.gif)


## Configuration

- Environment parameters can be set in `src/env_builders/constants.py`.

## File Overview

- `src/train.py`: Entry point for training and testing.
- `src/algorithms/abstract_rl_algorithm.py`: Abstract base class for RL algorithms.
- `src/algorithms/monte_carlo.py`: Monte Carlo implementation.
- `src/algorithms/q_learning.py`: Q-Learning implementation.
- `src/env_builders/cliff_walking.py`: Environment wrapper.

## Requirements

- Python 3.7+
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- pygame

Install all requirements with:

```sh
pip install -r requirements.txt
```

## License

MIT License

---

*This project is for educational and benchmarking purposes.*