from env_builders.cliff_walking import CliffWalking
from algorithms.monte_carlo import MonteCarlo
from algorithms.q_learning import QLearning

if __name__ == "__main__":
    for Algorithm, train_steps in [(MonteCarlo, 30000), (QLearning, 3000)]:
        env = CliffWalking()
        algorithm = Algorithm(env)
        env.info()
        algorithm.train(train_steps)
        algorithm.test()
        env.close()