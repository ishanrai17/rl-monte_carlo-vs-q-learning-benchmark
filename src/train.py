from env_builders.cliff_walking import CliffWalking
from algorithms.monte_carlo import MonteCarlo
from algorithms.q_learning import QLearning

if __name__ == "__main__":
    env = CliffWalking()
    algorithm = QLearning(env)
    env.info()
    algorithm.train(20000)
    algorithm.test()
