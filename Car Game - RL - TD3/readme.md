
# Autonomous Car Navigation in Simulated Environment

## Project Description
This project focuses on developing an autonomous car navigation system within a simulated environment. The environment is represented using a binary map, distinguishing between the road and non-road areas. The objective is to navigate the car efficiently and safely to a destination, primarily staying on the road.

## Environment
The environment is a binary representation:
- `0` indicates road (navigable area).
- `1` represents non-road areas (such as obstacles or off-road terrain).

## Feature Representation
Features used for navigation include:
- **Three Signals**: Sensor data about immediate surroundings.
- **Orientation**: The car's current orientation.
- **Sand Density**: Indicates the presence of off-road terrain.
- **Distance to Goal**: Euclidean distance to the destination.
- **Angle to Goal**: Angle between car's orientation and direction to the goal.
- **Road Following Performance**: Metric evaluating how well the car stays on the road.
- **Vector Field**: For path planning, indicating direction towards the goal.
- **Obstacle Field**: Directions away from nearby obstacles.

## Reward Structure
The reward system incentivizes on-road driving, quick destination reach, and obstacle avoidance:
```python
    last_reward = 0
    if sand[int(self.car.x), int(self.car.y)] > 0:
        self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
        last_reward -= 2
    else: # otherwise
        self.car.velocity = Vector(2.0, 0).rotate(self.car.angle)
        last_reward += 0.5

    if is_on_road:
        on_road_counter += 1
        off_road_counter = 0
        last_reward += 0.01 * on_road_counter
    else:
        off_road_counter += 1
        on_road_counter = 0
        last_reward += -0.05 * off_road_counter 

    if distance < last_distance:
        last_reward += 0.6  # closer to destination
        last_reward += 0.1 / (distance + 1)

    if sand_density > 0.7:
        last_reward -= 0.5

    if self.car.x < 5:
        self.car.x = 5
        last_reward -= 1 if not is_on_road else 0
    if self.car.x > self.width - 5:
        self.car.x = self.width - 5
        last_reward -= 1 if not is_on_road else 0
    if self.car.y < 5:
        self.car.y = 5
        last_reward -= 1 if not is_on_road else 0
    if self.car.y > self.height - 5:
        self.car.y = self.height - 5
        last_reward -= 1 if not is_on_road else 0

    if distance < 25:
        last_reward += 6.0
        swap = (swap + 1) % (len(fixed_destinations))
        goal_x, goal_y = fixed_destinations[swap]
        self.goal.x, self.goal.y,  = goal_x, goal_y
    ```

## TD3 Network Architecture
![image](https://github.com/Delve-ERAV1/Reinforcement-Learning/assets/11761529/34e02f57-e63c-4a2d-9202-a391194c7e3f)

The project uses the Twin Delayed Deep Deterministic policy gradient (TD3) algorithm:
- **Actor Network**: Decides the best action in the current state.
- **Critic Network**: Evaluates actions and guides the Actor network.

```python
class Actor(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, p)
    self.layer_2 = nn.Linear(p, p*8)
    self.layer_3 = nn.Linear(p*8, p*8)
    self.layer_4 = nn.Linear(p*8, action_dim)

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = F.relu(self.layer_3(x))
    return  5 * torch.tanh(self.layer_4(x))
  

class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, p)
    self.layer_2 = nn.Linear(p, p*8)
    self.layer_3 = nn.Linear(p*8, p*8)
    self.layer_4 = nn.Linear(p*8, action_dim)
    # Defining the second Critic neural network
    self.layer_11 = nn.Linear(state_dim + action_dim, p)
    self.layer_12 = nn.Linear(p, p*8)
    self.layer_13 = nn.Linear(p*8, p*8)
    self.layer_14 = nn.Linear(p*8, action_dim)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = F.relu(self.layer_3(x1))
    x1 = self.layer_4(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_11(xu))
    x2 = F.relu(self.layer_12(x2))
    x2 = F.relu(self.layer_13(x2))
    x2 = self.layer_14(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = F.relu(self.layer_3(x1))
    x1 = self.layer_4(x1)
    return x1```

## Speed Selection and Control
Initially, the speed of the car was manually selected based on whether it was on or off the road:
- `0.5` when off-road to simulate cautious driving.
- `2.0` when on-road for optimal speed.

A subsequent implementation extended the network to also predict the car's speed, allowing the model to dynamically choose both steering and speed based on the environment and situation.

### TD3 for Speed Prediction

```python
class Actor(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, p)
    self.layer_2 = nn.Linear(p, p//2)
    self.layer_3 = nn.Linear(p//2, action_dim)

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.layer_3(x)

    steering = 5 * torch.tanh(x[:, 0])
    speed = 2 * torch.sigmoid(x[:, 1])
    
    return torch.cat([steering.unsqueeze(1), speed.unsqueeze(1)], dim=1)
```

The TD3 network architecture was adapted to output two values:
- One for steering angle (range: -5 to +5).
- One for speed (range: 0 to 2).

This approach gives the car more autonomy and adaptability in navigating the environment, enhancing the model's ability to make context-aware decisions.

## Use of Convolutions
For feature representation, an `nxn` grid around the car's location is extracted, utilizing convolutional techniques for effective spatial data processing.

## Installation and Running the Simulation
Instructions on setting up and running the simulation.

## Dependencies
List of dependencies and installation instructions.

## Simulation Video
[![IMAGE ALT TEXT HERE](https://github.com/Delve-ERAV1/Reinforcement-Learning/assets/11761529/24454448-8ff2-4360-b076-ed3226bed4ec)](https://youtu.be/CR5v3uaJgHA)

## References
* www.udemy.com%2Fcourse%2Fdeep-reinforcement-learning%2F
