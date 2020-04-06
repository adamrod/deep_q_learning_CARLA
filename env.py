import sys
import os
sys.path.append(os.getenv('CARLA_api_egg'))
import carla
import time
import numpy as np
import cv2


IMG_WIDTH = 30
IMG_HEIGHT = 30
THROTTLE = 0.6


class CarlaEnv:
    def __init__(self, spawn_index=0, action_size=3):
        self.actor_list = []
        self.collision_history = []
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.bp_library = self.world.get_blueprint_library()
        self.action_size = action_size
        self.last_road_side = None

        self._create_blueprints()
        self._create_actors(spawn_index)

    def reset(self, spawn_index=0):
        self.ss_camera_image = None
        self.distance = None

        self.vehicle.set_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        time.sleep(0.4)
        
        self.vehicle.set_transform(self.spawn_points[spawn_index])
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=THROTTLE, steer=0))
        while self._get_kmh() < 10:
            time.sleep(0.1)
    
        while self.ss_camera_image is None:
            time.sleep(0.1)

        self.collision_history = []

        return self.ss_camera_image

    def step(self, action):
        action /= ((self.action_size - 1) / 2)
        steer_value = action - 1.0
        self.vehicle.apply_control(carla.VehicleControl(throttle=THROTTLE, steer=steer_value))

        time.sleep(0.1)

        if len(self.collision_history) > 0:
            done = True
            reward = -1
        else:
            done = False
            max_state = IMG_WIDTH - 1
            optimal_state = max_state / 2
            normalised_penalty = abs(optimal_state - self.distance) / optimal_state
            reward = 0.1 - 0.1 * normalised_penalty**2
        
        return self.ss_camera_image, reward, done

    def close(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        
    def _create_blueprints(self):
        self.bp_vehicle = self.bp_library.find('vehicle.tesla.model3')

        self.bp_collision_sensor = self.bp_library.find('sensor.other.collision')

        self.bp_ss_camera = self.bp_library.find('sensor.camera.semantic_segmentation')
        self.bp_ss_camera.set_attribute('image_size_x', str(IMG_WIDTH))
        self.bp_ss_camera.set_attribute('image_size_y', str(IMG_HEIGHT))

    def _create_actors(self, spawn_index):
        self.spawn_point = self.spawn_points[spawn_index]

        self.vehicle = self.world.spawn_actor(self.bp_vehicle, self.spawn_point)
        self.actor_list.append(self.vehicle)

        self.collision_sensor = self.world.spawn_actor(self.bp_collision_sensor, carla.Transform(carla.Location(x=1, z=2.2)), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._update_collisions(event))
        self.actor_list.append(self.collision_sensor)

        self.ss_camera = self.world.spawn_actor(self.bp_ss_camera, carla.Transform(carla.Location(x=1, z=2.2)), attach_to=self.vehicle)
        self.ss_camera.listen(lambda image: self._update_ss_image(image))
        self.actor_list.append(self.ss_camera)

    def _update_ss_image(self, image):
        image = np.array(image.raw_data)
        image = image.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        image = image[int(IMG_HEIGHT//1.7):, :, 2]
        image[(image != 6) & (image != 7)] = 0
        image[(image == 6) | (image == 7)] = 1
        image = np.expand_dims(image, axis=2)
        self.ss_camera_image = image

        # Calculate distance to the middle of the road
        line = image[IMG_HEIGHT // 6]
        line_sum = sum(line)
        if line_sum == 0:
            if self.last_road_side == "left":
                self.distance = 0
            else:
                self.distance = len(line) - 1
        else:
            left_sum = 0
            for i in range(len(line)):
                left_sum += line[i]
                if left_sum >= line_sum / 2:
                    self.distance = i
                    break
            if self.distance < len(line) / 2:
                self.last_road_side = "left"
            else:
                self.last_road_side = "right"

    def _update_collisions(self, event):
        self.collision_history.append(event)

    def _get_kmh(self):
        v = self.vehicle.get_velocity()
        return int(3.6 * ((v.x**2 + v.y**2)**(0.5)))


if __name__ == '__main__':
    # For testing purpose
    try:
        env = CarlaEnv()
        for _ in range(10):
            state = env.reset()
            for _ in range(10):  
                state, reward, done = env.step(1)
                time.sleep(0.1)
    finally:
        env.close()
