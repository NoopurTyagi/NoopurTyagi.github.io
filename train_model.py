# train_model.py
import gym
import numpy as np
from stable_baselines3 import PPO

# Define waypoints
waypoints = [
    {"id": 1, "name": "Entrance", "x": -8, "y": 0, "z": -3, "connected": [2]},
    {"id": 2, "name": "Corridor Intersection", "x": -6, "y": 0, "z": -3, "connected": [1, 3, 4, 5, 6, 7, 8]},
    {"id": 3, "name": "Faculty Room 03", "x": -6, "y": 0, "z": -10, "connected": [2]},
    {"id": 4, "name": "Faculty Room 004", "x": -2, "y": 0, "z": 2, "connected": [2]},
    {"id": 5, "name": "International Office", "x": -6, "y": 0, "z": 2, "connected": [2]},
    {"id": 6, "name": "Restroom 1", "x": -4, "y": 0, "z": -10, "connected": [2]},
    {"id": 7, "name": "Restroom 2", "x": -2, "y": 0, "z": -10, "connected": [2]},
    {"id": 8, "name": "Pantry", "x": -1, "y": 0, "z": -7, "connected": [2]},
    {"id": 9, "name": "Ground Floor Stairs", "x": -1, "y": 2, "z": -5, "connected": [2, 10]},
    {"id": 10, "name": "Ground Floor Stairs End", "x": 0, "y": 4, "z": -6, "connected": [9, 11]},
    {"id": 11, "name": "First Floor Stairs", "x": 0, "y": 4, "z": -4, "connected": [10, 12]},
    {"id": 12, "name": "Corridor", "x": -5, "y": 3.5, "z": -4, "connected": [11, 13]},
    {"id": 13, "name": "First Floor Corridor Intersection", "x": -6, "y": 3.5, "z": 3, "connected": [12, 14]},
    {"id": 14, "name": "LH2", "x": -1, "y": 3.5, "z": -1, "connected": [13]},
    {"id": 15, "name": "LH1", "x": -1, "y": 3.5, "z": -1, "connected": [14]}
]

# Ensure bidirectional connections
for waypoint in waypoints:
    for conn in waypoint["connected"]:
        neighbor = next((w for w in waypoints if w["id"] == conn), None)
        if neighbor and waypoint["id"] not in neighbor["connected"]:
            neighbor["connected"].append(waypoint["id"])

# Define the Gym environment for pathfinding
class IndoorNavEnv(gym.Env):
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.action_space = gym.spaces.Discrete(len(waypoints))
        self.observation_space = gym.spaces.Box(low=0, high=len(waypoints), shape=(1,))
        self.reset()

    def reset(self):
        self.current_waypoint = np.random.choice(len(self.waypoints))
        return np.array([self.current_waypoint])

    def step(self, action):
        next_waypoint = action
        reward = -np.linalg.norm(np.array([self.waypoints[next_waypoint]["x"], self.waypoints[next_waypoint]["y"], self.waypoints[next_waypoint]["z"]]) -
                                 np.array([self.waypoints[self.current_waypoint]["x"], self.waypoints[self.current_waypoint]["y"], self.waypoints[self.current_waypoint]["z"]]))
        self.current_waypoint = next_waypoint
        done = (self.current_waypoint == end_waypoint)
        return np.array([self.current_waypoint]), reward, done, {}

env = IndoorNavEnv(waypoints)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

model.save("indoor_nav_model")
