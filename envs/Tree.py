import gym
from gym import spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

class Tree(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, depth=3, max_steps=100):
        super(Tree, self).__init__()
        self.depth = depth
        self.max_steps = max_steps
        self.tree = self._create_binary_tree(self.depth)
        self.leaf_nodes = [node for node, degree in self.tree.degree() if degree == 1 and self.tree.nodes[node]['depth'] == depth]
        self.num_leaf_nodes = len(self.leaf_nodes)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Discrete(2 ** (self.depth + 1) - 1)  # Total number of nodes in a binary tree of given depth

        self.current_node = 0
        self.steps_taken = 0

    def _create_binary_tree(self, depth):
        G = nx.Graph()
        nodes = [(0, 0)]  # (current node, depth)
        G.add_node(0, depth=0)

        node_id = 1

        while nodes:
            current_node, current_depth = nodes.pop(0)
            if current_depth < depth:
                left_child = node_id
                right_child = node_id + 1
                G.add_node(left_child, depth=current_depth + 1)
                G.add_node(right_child, depth=current_depth + 1)
                G.add_edge(current_node, left_child)
                G.add_edge(current_node, right_child)
                nodes.append((left_child, current_depth + 1))
                nodes.append((right_child, current_depth + 1))
                node_id += 2

        return G

    def reset(self):
        self.current_node = 0
        self.steps_taken = 0
        return self.current_node

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        neighbors = list(self.tree.neighbors(self.current_node))
        if len(neighbors) > 1:
            next_node = neighbors[action]
        else:
            next_node = neighbors[0]
        self.current_node = next_node

        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps
        reward = 0

        return self.current_node, reward, done, {}

    def render(self, mode='human'):
        pos = nx.multipartite_layout(self.tree, subset_key="depth")
        nx.draw(self.tree, pos, with_labels=True, node_size=500,
                node_color="skyblue", font_size=10, font_color="black",
                font_weight="bold", edge_color="gray")
        plt.show()

    def close(self):
        pass

    def get_dataset(self, num_episodes=2):
        max_steps = self.max_steps

        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []

        for _ in range(num_episodes):
            obs = self.reset()
            for _ in range(max_steps):
                action = self.action_space.sample()
                next_obs, reward, done, _ = self.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                terminals.append(done)
                timeouts.append(done and self.steps_taken == self.max_steps)

                obs = next_obs

                if done:
                    break

        dataset = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'terminals': np.array(terminals),
            'timeouts': np.array(timeouts)
        }

        return dataset

# Register the environment
gym.envs.registration.register(
    id='BinaryTree-v0',
    entry_point=Tree,
)

# Usage example
if __name__ == "__main__":
    env = gym.make('BinaryTree-v0', depth=3, max_steps=10)
    dataset = env.get_dataset()

    print(f"Observations: {dataset['observations']}")
    print(f"Actions: {dataset['actions']}")
    print(f"Rewards: {dataset['rewards']}")
    print(f"Terminals: {dataset['terminals']}")
    print(f"Timeouts: {dataset['timeouts']}")
