from .abstract_rl_algorithm import AbstractRLAlgorithm

class MonteCarlo(AbstractRLAlgorithm):
    def _execute_episode(self, update=True):
        current_state, _ = self.env.reset()
        total_reward = 0
        complete = False
        step_count = 0
        episode = []

        while not complete and step_count < self.env.maximum_steps:
            self.env.render()
            action = self.choose_action(current_state)
            next_state, reward, complete = self.env.perform(action)
            episode.append((current_state, action, reward))
            current_state = next_state
            total_reward += reward
            step_count += 1

        self.env.render()

        if update:
            self._update(episode)

        if self.episode_num % 100 == 0:
            print(f"Episode {self.episode_num}: Total Reward: {total_reward}, Steps: {step_count}, Epsilon: {self.epsilon:.4f}, Alpha: {self.alpha:.4f}")

    def _update(self, episode):
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                self.Q[state][action] += (G - self.Q[state][action]) * self.alpha