import gym as gym
import random
from time import sleep
from .constants import MAXIMUM_STEPS, RENDER_MODE

ENV_NAME = "CliffWalking-v0"

class CliffWalking:
    def __init__(self, render_mode=RENDER_MODE):
        self.env = gym.make(ENV_NAME, render_mode=render_mode)
        self.maximum_steps = MAXIMUM_STEPS
        self.complete_count = 0

        # Initialize Q(s,a) and a counter of first visits
        # self.Q = [[0]*self.env.action_space.n for _ in range(self.env.observation_space.n)]
        # self.returns_count = [[0]*self.env.action_space.n for _ in range(self.env.observation_space.n)]

    def set_render_mode(self, render_mode):
        self.env = gym.make(ENV_NAME, render_mode=render_mode)

    def info(self):
        print("CliffWalking Environment")
        print("Action space:", self.env.action_space)
        print("Available actions:", list(range(self.env.action_space.n)))
        print("Observation space:", self.env.observation_space)

    def perform(self, action):
        state, reward, complete, _, info = self.env.step(action)
        return state, reward, complete

    def random_action(self):
        return self.env.action_space.sample()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def reset(self):
        self.cliff_count = 0
        self.win_count = 0
        return self.env.reset()

    def build_Q_table(self):
        return [[0]*self.env.action_space.n for _ in range(self.env.observation_space.n)]