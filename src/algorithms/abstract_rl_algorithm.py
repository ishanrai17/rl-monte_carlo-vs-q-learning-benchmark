from .constants import GAMMA, EPSILON, ALPHA, DECAY_RATE
import random
from abc import ABC, abstractmethod

class AbstractRLAlgorithm(ABC):
    def __init__(self, env):
        self.env = env
        self.Q = self.env.build_Q_table()
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.alpha = ALPHA
        self.decay_rate = DECAY_RATE
        self.episode_num = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.random_action()
        else:
            qsa = self.Q[state]
            return qsa.index(max(qsa))

    @abstractmethod
    def _execute_episode(self, update=True):
        pass

    def train(self, num_episodes):
        try:
            r = 0
            for self.episode_num in range(num_episodes):
                r += self._execute_episode()

                if self.episode_num % 1000 == 0:
                    print(f"Average reward over last 1000 episodes: {r / 1000:.2f}")
                    r = 0

                self.epsilon = max(0.1, self.epsilon * self.decay_rate)
                self.alpha = ALPHA * (0.9999 ** self.episode_num)
                self.alpha = max(0.01, self.alpha)
            self.env.close()
        except KeyboardInterrupt:
            print("Training interrupted.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print("Training finished.")
            self.env.close()

    def test(self):
        try:
            self.env.set_render_mode('human')
            self._execute_episode(update=False)
            self.env.set_render_mode(None)
        except KeyboardInterrupt:
            print("Testing interrupted.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print("Testing finished.")
            self.env.set_render_mode(None)
            self.env.close()