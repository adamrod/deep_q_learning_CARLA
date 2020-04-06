import sys
import os
sys.path.append(os.getenv('CARLA_api_egg'))
import carla


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
world.set_weather(carla.WeatherParameters.ClearNoon)

settings = world.get_settings()
settings.fixed_delta_seconds = 0.02
world.apply_settings(settings)
