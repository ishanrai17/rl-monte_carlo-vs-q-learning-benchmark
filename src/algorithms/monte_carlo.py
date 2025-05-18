from .abstract_rl_algorithm import AbstractRLAlgorithm

GAMMA = 1
ALPHA = 0.1
EPSILON = 0.99
DECAY_RATE = 0.99995

class MonteCarlo(AbstractRLAlgorithm):
    def __init__(self, env):
        super().__init__(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, decay_rate=DECAY_RATE)

    def _execute_episode(self, update=True):
        current_state, _ = self.env.reset()
        total_reward = 0
        complete = False
        episode = []

        while not complete:
            self.env.render()
            action = self.choose_action(current_state)
            next_state, reward, complete, step_count = self.env.perform(action)
            episode.append((current_state, action, reward))
            current_state = next_state
            total_reward += reward

        self.env.render()

        if update:
            self._update(episode)

        if self.episode_num % 1000 == 0:
            print(f"Episode {self.episode_num}: Total Reward: {total_reward}, Steps: {step_count} Epsilon: {self.epsilon:.4f}, Alpha: {self.alpha:.4f}")

        return total_reward

    def _update(self, episode):
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                self.Q[state][action] += (G - self.Q[state][action]) * self.alpha