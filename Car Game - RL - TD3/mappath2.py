# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import heapq


# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from multiprocessing import Pool


# Importing the Dqn object from our AI in ai.py
from aipath2 import TD3
from skimage.transform import rescale, resize

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', True)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = TD3(9, 1, [(-5, 5), (0.5, 2)])
#action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")


img = PILImage.open("./images/mask.png").convert('L')
sand = np.asarray(img)/255

down_sample = 10


def get_random_road_point():
    while True:
        # Randomly select a point within the map boundaries
        x = randint(0, sand.shape[0] - 1)
        y = randint(0, sand.shape[1] - 1)

        # Check if the selected point is on the road (sand value of 0)
        if sand[x, y] == 0:
            return x, y

def downsample_map(map_data, scale_factor):
    """
    Downsample the map by the given scale factor using average pooling.

    :param map_data: 2D numpy array representing the original map.
    :param scale_factor: Factor by which to downsample the map.
    :return: Downsampled map.
    """
    # Using resize for average pooling effect
    return resize(map_data, (map_data.shape[0] // scale_factor, map_data.shape[1] // scale_factor), 
                  anti_aliasing=True, order=1, mode='reflect', preserve_range=True)


# Downsample the map by a factor of 4
downsampled_map = downsample_map(sand, down_sample)
vector_field = np.zeros((downsampled_map.shape[0], downsampled_map.shape[1], 2), dtype=np.float32)


def heuristic(a, b):
    """
    Heuristic function for A* algorithm, using Euclidean distance.

    :param a: Tuple representing the current grid cell.
    :param b: Tuple representing the target grid cell.
    :return: Euclidean distance between a and b.
    """
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def a_star(array, start, goal):
    """
    A* pathfinding algorithm on a 2D grid.

    :param array: 2D numpy array representing the map.
    :param start: Tuple representing the start coordinates.
    :param goal: Tuple representing the goal coordinates.
    :return: List of tuples as a path from the start to the goal.
    """
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 4-way movement
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_set = []

    heapq.heappush(open_set, (fscore[start], start))
    
    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (fscore[neighbor], neighbor))
                
    return []

def compute_vector_field(downsampled_map, destination, location):
    """
    Compute the vector field for a downsampled map, with vectors pointing from each road cell
    to the next step on the shortest path to the destination.

    :param downsampled_map: 2D numpy array representing the downsampled map.
    :param destination: Tuple of (x, y) coordinates for the destination in the downsampled map.
    :return: 2D numpy array representing the vector field.
    """

    for i in range(downsampled_map.shape[0]):
        for j in range(downsampled_map.shape[1]):
            if downsampled_map[i, j] < 0.5:  # Assuming road cells are represented by values < 0.5
                path = a_star(downsampled_map, (i, j), destination)
                if len(path) > 1:  # If there is a next step in the path
                    direction = np.array(path[1]) - np.array(path[0])
                    norm = np.linalg.norm(direction)
                    if norm != 0:
                        direction = direction / norm  # Normalize the direction vector
                    vector_field[i, j] = direction
                else:
                    vector_field[i, j] = [0, 0]  # Zero vector for cells with no path to destination

    return(vector_field[int(location[0]), int(location[1])])


def compute_vector_at_point(downsampled_map, i, j, destination):
    if downsampled_map[i, j] < 0.5:  # Road
        path = a_star(downsampled_map, (i, j), destination)
        if len(path) > 1:
            direction = np.array(path[1]) - np.array(path[0])
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction = direction / norm
            return i, j, direction
    return i, j, [0, 0]

def compute_vector_field_multiprocessing(downsampled_map, destination, location):
    
    # Create tasks for each point in the downsampled map
    tasks = [(downsampled_map, i, j, destination) for i in range(downsampled_map.shape[0]) 
                                                     for j in range(downsampled_map.shape[1])]

    # Create a pool of worker processes
    with Pool() as pool:
        results = pool.starmap(compute_vector_at_point, tasks)

    # Fill in the vector field with results from worker processes
    for i, j, vector in results:
        vector_field[i, j] = vector

    return(vector_field[int(location[0]), int(location[1])])

# Initializing the map
first_update = True
fixed_destinations = [get_random_road_point() for _ in range(3)]
goal_x, goal_y = fixed_destinations[0]

def init():
    global sand
    global first_update
    #sand = np.zeros((longueur,largeur))
    
    #goal_x = 1420
    #goal_y = 622
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0


# Creating the car class
class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    road_following_history = [] 
    stagnation_counter = 0

    def get_relative_goal_position(self, goal_x, goal_y):
        # vector from the car to the goal
        vector_to_goal = Vector(goal_x - self.x, goal_y - self.y)

        # angle between the car's direction and the vector to the goal
        car_direction = Vector(*self.velocity).normalize()
        angle_to_goal = vector_to_goal.angle(car_direction)

        # if the goal is to the left or right of the car
        goal_on_left = angle_to_goal > 0
        goal_on_right = angle_to_goal < 0

        # Determine if the goal is in front of or behind the car
        car_to_goal_direction = vector_to_goal.normalize()
        forward_direction = Vector(1, 0).rotate(self.angle).normalize()
        goal_in_front = car_to_goal_direction.dot(forward_direction) > 0

        return int(goal_on_left), int(goal_on_right), int(goal_in_front)

    def calculate_goal_direction(self, goal_x, goal_y):
        # vector from the car to the goal
        vector_to_goal = Vector(goal_x - self.x, goal_y - self.y)

        # distance to the goal
        distance_to_goal = vector_to_goal.length()
        normalized_distance = distance_to_goal / max_distance

        # Calculate the angle between the car's direction and the vector to the goal
        car_direction = Vector(*self.velocity).normalize()
        angle_to_goal = vector_to_goal.angle(car_direction)
        normalized_angle = angle_to_goal / 180.0


        return normalized_angle, normalized_distance


    def update_road_following_history(self, is_on_road):
        self.road_following_history.append(is_on_road)
        if len(self.road_following_history) > 1000:
            del self.road_following_history[0]

    def get_road_following_performance(self):
        return(sum(self.road_following_history) / len(self.road_following_history) if self.road_following_history else 0)


    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        self.check_sensor_bounds()

    def check_sensor_bounds(self):
        # Check if sensors are out of bounds and adjust their signals
        if self.sensor1_x > longueur-10 or self.sensor1_x < 10 or self.sensor1_y > largeur-10 or self.sensor1_y < 10:
            self.signal1 = 10.
        if self.sensor2_x > longueur-10 or self.sensor2_x < 10 or self.sensor2_y > largeur-10 or self.sensor2_y < 10:
            self.signal2 = 10.
        if self.sensor3_x > longueur-10 or self.sensor3_x < 10 or self.sensor3_y > largeur-10 or self.sensor3_y < 10:
            self.signal3 = 10.

    def calculate_sand_density(self, area_size=10):
        x_min = max(int(self.x) - area_size, 0)
        x_max = min(int(self.x) + area_size, sand.shape[1])
        y_min = max(int(self.y) - area_size, 0)
        y_max = min(int(self.y) + area_size, sand.shape[0])

        area = sand[y_min:y_max, x_min:x_max]
        return(np.mean(area) if area.size > 0 else 1)

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Goal(Widget): 
    pass
# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    goal = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global max_distance

        longueur = self.width
        largeur = self.height
        max_distance = np.sqrt(longueur**2 + largeur**2)
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.

        is_on_road = sand[int(self.car.x), int(self.car.y)] == 0
        self.car.update_road_following_history(is_on_road)

        road_following_performance = self.car.get_road_following_performance()

        sand_density = round(self.car.calculate_sand_density(), 3)

        angle_to_goal, distance_to_goal = self.car.calculate_goal_direction(goal_x, goal_y)
        angle_to_goal, distance_to_goal = round(angle_to_goal, 3), round(distance_to_goal, 3) 

        downsampled_destination = (goal_x // down_sample, goal_y // down_sample)  # Adjust destination for downsampled map
        v_field = compute_vector_field_multiprocessing(downsampled_map, downsampled_destination, [self.car.x//down_sample, self.car.y//down_sample])


        print(v_field)
        print(type(v_field))
        print(len(v_field)) 

        #v_field = self.car.calculate_vector_field([goal_x, goal_y]) 
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, 
                       orientation, -orientation, 
                       road_following_performance, distance_to_goal, angle_to_goal, sand_density,
                       #self.car.velocity[0], self.car.velocity[1]
                       ]
        print(f"RFP: {road_following_performance} d2g: {distance_to_goal} a2g: {angle_to_goal} sd: {sand_density}")
        #action, speed = brain.update(last_reward, last_signal)
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action #action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        self.goal.center = Vector(goal_x, goal_y)

        #self.car.velocity = Vector(speed, 0).rotate(self.car.angle)

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            #print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            
            last_reward = -2
        else: # otherwise
            self.car.velocity = Vector(2.0, 0).rotate(self.car.angle)
            last_reward = 1
            #print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.6
            # else:
            #     last_reward = last_reward +(-0.2)

        if road_following_performance < 0.7:
            last_reward -= 2
        else:
            last_reward += 1

        if sand_density > 0.7:
            last_reward -= 2
        else:
            last_reward += 1

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -1
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -1
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -1

        if distance < 25:
            last_reward += 4.0
            swap = (swap + 1) % 3
            goal_x, goal_y = fixed_destinations[swap]
            self.goal.center = Vector(goal_x, goal_y)

        last_distance = distance
        self.goal.center = Vector(goal_x, goal_y)

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        # self.painter = MyPaintWidget()
        # clearbtn = Button(text = 'clear')
        # savebtn = Button(text = 'save', pos = (parent.width, 0))
        # loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        # clearbtn.bind(on_release = self.clear_canvas)
        # savebtn.bind(on_release = self.save)
        # loadbtn.bind(on_release = self.load)
        # parent.add_widget(self.painter)
        # parent.add_widget(clearbtn)
        # parent.add_widget(savebtn)
        # parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
