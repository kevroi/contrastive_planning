import gym
from gym import spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

class Tree(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, depth=3):
        super(Tree, self).__init__()
        self.depth = depth
        self.tree = self._create_binary_tree(self.depth)
        self.leaf_nodes = [node for node, degree in self.tree.degree() if degree == 1 and self.tree.nodes[node]['depth'] == depth]
        self.num_leaf_nodes = len(self.leaf_nodes)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Discrete(2 ** (self.depth + 1) - 1)  # Total number of nodes in a binary tree of given depth

        self.current_node = 0

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
        return self.current_node

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        next_node = list(self.tree.neighbors(self.current_node))[action]
        self.current_node = next_node

        done = self.current_node in self.leaf_nodes
        reward = 1 if done else 0

        return self.current_node, reward, done, {}

    def render(self, mode='human'):
        pos = nx.multipartite_layout(self.tree, subset_key="depth")
        nx.draw(self.tree, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray")
        plt.show()

    def close(self):
        pass

# Register the environment
gym.envs.registration.register(
    id='BinaryTree-v0',
    entry_point=Tree,
)

# Usage example
if __name__ == "__main__":
    env = gym.make('BinaryTree-v0', depth=3)
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}")
        if done:
            break
    env.render()
