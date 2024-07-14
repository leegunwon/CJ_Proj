import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import deque
from stacking_algorithm_sample import StackingAlgorithm, StackingMethod
from stacking_main import stacking_check_and_visualize, is_point_in_box, generate_boxes
from action_manager import ActionManager

# Hyperparameters
EPISODES = 1000
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Q-learning functions
def choose_action(state, policy_net, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(methods))
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()


def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = torch.tensor(batch_state, dtype=torch.float32)
    batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)
    batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
    batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)

    current_q_values = policy_net(batch_state).gather(1, batch_action)
    max_next_q_values = target_net(batch_next_state).max(1)[0].detach().unsqueeze(1)
    expected_q_values = batch_reward + (GAMMA * max_next_q_values * (1 - batch_done))

    loss = nn.MSELoss()(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main training function
def train_dqn(data_dir, cubic_range, num_small_boxes, num_large_boxes, episodes, alpha, gamma, epsilon_start,
              epsilon_end, epsilon_decay):
    input_dim = len(methods)  # Input dimension for the neural network
    output_dim = len(methods)  # Output dimension (number of actions)

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = epsilon_start

    for episode in range(episodes):
        # Generate random boxes
        boxes_small = generate_boxes(num_small_boxes, x_range=[200, 500], y_range=[200, 500], z_range=[200, 500])
        boxes_large = generate_boxes(num_large_boxes, x_range=[500, 700], y_range=[500, 700], z_range=[500, 700])
        boxes = boxes_small + boxes_large
        save_to_json(boxes, os.path.join(data_dir, f'boxes_episode_{episode}.json'))

        loaded_boxes = load_from_json(os.path.join(data_dir, f'boxes_episode_{episode}.json'))
        ActionManager = ActionManager(loaded_boxes, cubic_range)

        state = np.zeros(input_dim)
        done = False
        score = 0
        for box in loaded_boxes:
            action = choose_action(state, policy_net, epsilon)
            method = methods[action]
            placements = ActionManager.stack(method, box)

            if not placements:
                reward = -1
                next_state = state
                done = True
            else:
                result_file_name = f'episode_{episode}_method_{method.name}'
                stacking_rate, _ = stacking_check_and_visualize(placements, loaded_boxes, cubic_range, data_dir,
                                                                result_file_name)
                next_state = np.zeros(input_dim)
                next_state[action] = 1

                if stacking_rate > 0:
                    reward = stacking_rate
                    done = True
                else:
                    reward = 0
            score += reward
            memory.push(state, action, reward, next_state, done)
            state = next_state

            optimize_model(policy_net, target_net, memory, optimizer)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            output_string = "--------------------------------------------------\n" + \
                            f"stacking_rate: {stacking_rate}" + \
                            f"n_episode: {episode}, score : {score:.1f}, n_buffer : {memory.size()}, eps : {epsilon * 100:.1f}%"
            print(output_string)

        epsilon = max(epsilon_end, epsilon_decay * epsilon)


# Helper functions
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


# Train the DQN model
data_dir = "sample_json"
cubic_range = [1100, 1100, 1800]
num_small_boxes, num_large_boxes = 75, 5

methods = \
    [
        StackingMethod.PALLET_Z_EXCEED           ,
        StackingMethod.PALLET_X_EXCEED           ,
        StackingMethod.PALLET_Y_EXCEED           ,
        StackingMethod.PALLET_CENTER             ,
        StackingMethod.PALLET_CENTER_ROT         ,
        StackingMethod.PALLET_CORNER_XY_AXIS     ,
        StackingMethod.PALLET_CORNER_Z_AXIS      ,
        # ActionManager.PALLET_STACK_ALL
    ]

train_dqn(data_dir, cubic_range, num_small_boxes, num_large_boxes, EPISODES, ALPHA, GAMMA, EPSILON_START, EPSILON_END,
          EPSILON_DECAY)
