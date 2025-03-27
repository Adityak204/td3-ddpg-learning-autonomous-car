# Self Driving Car with TD3
from loguru import logger
import os
from datetime import datetime
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
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
from kivy.core.window import Window
import math
import torch
import numpy as np
from torch.autograd import Variable
import random

# Importing the TD3 object
from td3_ddpg.ai import TD3


def safe_float(value):
    """Convert any numeric value to a float that Kivy can handle"""
    if hasattr(value, "item"):  # For torch tensors
        return float(value.item())
    elif hasattr(value, "tolist"):  # For numpy arrays
        return float(value.tolist())
    return float(value)


# Setting up logging
def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"td3_training_{timestamp}.log")

    logger.remove()
    logger.add(
        lambda msg: print(msg),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}",
        colorize=True,
        level="INFO",
    )

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
    )

    logger.info(f"TD3 Training Log: {log_file}")
    return log_file


log_file = setup_logging()

# Configuration
Config.set("input", "mouse", "mouse,multitouch_on_demand")
Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1300")
Config.set("graphics", "height", "780")

# Global variables
last_x, last_y = 0, 0
n_points = length = 0
last_reward = 0
scores = []
im = CoreImage("./images/roads.jpg")

# Initialize TD3 agent
# State dimensions: [sensor1, sensor2, sensor3, orientation, -orientation] = 5
# Action dimensions: [steering, acceleration] = 2
brain = TD3(5, 2, 1.0)  # state_dim, action_dim, max_action
MAX_STEERING_ANGLE = 15  # degrees
MAX_ACCELERATION = 0.2
MIN_SPEED = 0.5
MAX_SPEED = 6.0

# Initializing the map
first_update = True


# Environment initialization
def init():
    global sand, goal_x, goal_y, first_update, swap
    img = PILImage.open("./images/roads_v.jpg").convert("L")
    sand = np.asarray(img)
    sand = np.where(sand < 255, 0, sand)
    sand = sand / 255  # Normalize to [0,1]
    goal_x = 230
    goal_y = 330
    first_update = False
    swap = 0


last_distance = 0


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

    def calculate_speed(self):
        """Calculate speed from velocity components without using .length()"""
        return math.sqrt(float(self.velocity_x) ** 2 + float(self.velocity_y) ** 2)

    def move(self, steering, acceleration):
        # Convert inputs to safe float values
        steering = float(steering)
        acceleration = float(acceleration)

        # Update position using safe float conversion
        new_x = float(self.x) + float(self.velocity_x)
        new_y = float(self.y) + float(self.velocity_y)
        self.pos = (new_x, new_y)

        # Update angle and rotation
        self.rotation = steering
        self.angle = (float(self.angle) + steering) % 360

        # Calculate current speed using our safe method
        current_speed = self.calculate_speed()

        # Calculate new speed with bounds
        new_speed = max(MIN_SPEED, min(current_speed + acceleration, MAX_SPEED))

        # Update velocity components
        angle_rad = math.radians(float(self.angle))
        self.velocity_x = float(new_speed * math.cos(angle_rad))
        self.velocity_y = float(new_speed * math.sin(angle_rad))

        self._update_sensors()
        self._get_sensor_readings()

    def _update_sensors(self):
        angle_rad = math.radians(safe_float(self.angle))
        offset = 30  # Sensor distance from car center

        self.sensor1_x = safe_float(self.x + offset * math.cos(angle_rad))
        self.sensor1_y = safe_float(self.y + offset * math.sin(angle_rad))
        self.sensor2_x = safe_float(
            self.x + offset * math.cos(angle_rad + math.radians(30))
        )
        self.sensor2_y = safe_float(
            self.y + offset * math.sin(angle_rad + math.radians(30))
        )
        self.sensor3_x = safe_float(
            self.x + offset * math.cos(angle_rad - math.radians(30))
        )
        self.sensor3_y = safe_float(
            self.y + offset * math.sin(angle_rad - math.radians(30))
        )

    def _get_sensor_readings(self):
        for i, (sensor_x, sensor_y) in enumerate(
            [
                (self.sensor1_x, self.sensor1_y),
                (self.sensor2_x, self.sensor2_y),
                (self.sensor3_x, self.sensor3_y),
            ]
        ):
            # Check boundaries
            if (
                sensor_x > longueur - 10
                or sensor_x < 10
                or sensor_y > largeur - 10
                or sensor_y < 10
            ):
                setattr(self, f"signal{i+1}", 1.0)
                continue

            # Get sand density in sensor area
            x1, x2 = max(0, int(sensor_x) - 10), min(longueur, int(sensor_x) + 10)
            y1, y2 = max(0, int(sensor_y) - 10), min(largeur, int(sensor_y) + 10)
            setattr(self, f"signal{i+1}", safe_float(np.mean(sand[x1:x2, y1:y2])))


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.pos = (1039, 394)
        self.car.center = self.center
        self.car.velocity_x = 3.0
        self.car.velocity_y = 0.0

    def update(self, dt):
        global brain, last_reward, scores, last_distance, goal_x, goal_y, longueur, largeur, swap

        longueur = self.width
        largeur = self.height

        if first_update:
            init()

        # Calculate orientation to goal (normalized to [-1,1])
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = math.atan2(yy, xx) - math.atan2(
            self.car.velocity_y, self.car.velocity_x
        )
        orientation = math.degrees(orientation) / 180.0

        # Normalize state inputs to [-1,1] range
        state = [
            float(self.car.signal1 * 2 - 1),  # Explicit float conversion
            float(self.car.signal2 * 2 - 1),
            float(self.car.signal3 * 2 - 1),
            float(np.clip(orientation, -1, 1)),
            float(np.clip(-orientation, -1, 1)),
        ]
        state = np.array(state, dtype=np.float32)  # Convert to numpy array

        # Get action from TD3 (steering and acceleration)
        action = brain.update(last_reward, state)

        # Scale actions to physical values
        steering = float(action[0] * MAX_STEERING_ANGLE)  # [-15,15] degrees
        acceleration = float(action[1] * MAX_ACCELERATION)  # [0,0.2] acceleration

        self.car.move(steering, acceleration)

        # Calculate distance to goal
        distance = math.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)

        # Update sensor visualization
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Ensure car stays within bounds
        self.car.x = safe_float(np.clip(float(self.car.x), 25, float(longueur) - 25))
        self.car.y = safe_float(np.clip(float(self.car.y), 25, float(largeur) - 25))

        # Improved reward shaping
        if sand[int(self.car.x), int(self.car.y)] > 0.5:  # Off-road
            last_reward = -2.0  # Penalty for being off-road
            # Reduce speed when off-road
            angle_rad = math.radians(float(self.car.angle))
            self.car.velocity_x = float(MIN_SPEED * math.cos(angle_rad))
            self.car.velocity_y = float(MIN_SPEED * math.sin(angle_rad))
        else:  # On-road
            speed_reward = self.car.calculate_speed() / MAX_SPEED  # [0,1]
            progress_reward = (
                last_distance - distance
            ) * 0.1  # Reward for getting closer

            last_reward = 0.1 * speed_reward + progress_reward

            # Maintain normal speed
            angle_rad = math.radians(float(self.car.angle))
            self.car.velocity_x = float(3.0 * math.cos(angle_rad))
            self.car.velocity_y = float(3.0 * math.sin(angle_rad))

        # Boundary penalties
        # if (
        #     self.car.x <= 25
        #     or self.car.x >= longueur - 25
        #     or self.car.y <= 25
        #     or self.car.y >= largeur - 25
        # ):
        #     last_reward = -5.0
        if self.car.x <= 25:
            self.car.x = 25
            last_reward = -5.0
        if self.car.x >= longueur - 25:
            self.car.x = longueur - 25
            last_reward = -5.0
        if self.car.y <= 25:
            self.car.y = 25
            last_reward = -5.0
        if self.car.y >= largeur - 25:
            self.car.y = largeur - 25
            last_reward = -5.0

        # Log detailed information
        logger.debug(
            f"State: {state}, Action: {action}, "
            f"Reward: {last_reward:.2f}, Distance: {distance:.1f}, "
            f"Speed: {self.car.calculate_speed():.2f}, "
            f"Position: ({self.car.x:.1f}, {self.car.y:.1f})"
        )

        # Goal reached
        if distance < 25:
            last_reward = 10.0  # Large reward for reaching goal
            if swap == 1:
                goal_x, goal_y = 974, 237
                swap = 0
            else:
                goal_x, goal_y = 810, 473
                swap = 1

        scores.append(brain.score())
        last_distance = distance


class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        return parent

    def save(self, obj):
        logger.info("Saving TD3 model...")
        brain.save()
        plt.figure(figsize=(10, 5))
        plt.plot(scores)
        plt.title("TD3 Training Progress")
        plt.xlabel("Timesteps")
        plt.ylabel("Average Reward")
        plt.savefig("./training_progress.png")
        plt.close()

    def load(self, obj):
        logger.info("Loading TD3 model...")
        brain.load()


if __name__ == "__main__":
    CarApp().run()
