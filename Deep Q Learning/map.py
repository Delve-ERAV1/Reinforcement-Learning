# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

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


# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', True)
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '667')


# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(9, 3, 0.9)
action2rotation = [0, 5, -5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

img = PILImage.open("./images/mask.png").convert('L')
sand = np.asarray(img)/255

# textureMask = CoreImage(source="./kivytest/simplemask1.png")

def get_random_road_point():
    while True:
        # Randomly select a point within the map boundaries
        x = randint(0, sand.shape[0] - 1)
        y = randint(0, sand.shape[1] - 1)

        # Check if the selected point is on the road (sand value of 0)
        if sand[x, y] == 0:
            return x, y

# Initializing the map
first_update = True

fixed_destinations = [get_random_road_point() for _ in range(3)]
goal_x, goal_y = fixed_destinations[0]

def init():
    global first_update
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
        sensor_distance = 30
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(sensor_distance, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(sensor_distance, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(sensor_distance, 0).rotate((self.angle-30)%360) + self.pos
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

        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, 
                       orientation, -orientation, 
                       road_following_performance, distance_to_goal, angle_to_goal, sand_density
                       ]

        print(f"RFP:{road_following_performance} - sd:{sand_density} - d2g:{distance_to_goal} - angle:{angle_to_goal} - stag:{self.car.stagnation_counter}")

        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        self.car.move(action2rotation[action])
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        self.goal.center = Vector(goal_x, goal_y)

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.7, 0).rotate(self.car.angle)
            
            last_reward = -2
        else: # otherwise
            last_reward = -0.2
            self.car.velocity = Vector(1.3, 0).rotate(self.car.angle)
            if distance < last_distance:
                last_reward = 0.1
                self.car.stagnation_counter = 0
            else:
                self.car.stagnation_counter += 1

        ## Penalize stagnation
        if self.car.stagnation_counter > 100:  # Define a threshold
            last_reward -= 0.3

        if road_following_performance > 0.8:
            last_reward += 0.1
        elif road_following_performance < 0.3: # todo: tune this
            last_reward += -0.1

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -1.0 #-1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -1.0 #-1
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -1.0 #-1
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -1.0 #-1

        if distance < 25:
            last_reward += 2.0
            swap = (swap + 1) % 3
            goal_x, goal_y = fixed_destinations[swap]
            self.goal.center = Vector(goal_x, goal_y)

        last_distance = distance

        print(f"Reward - {last_reward}")

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
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
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
