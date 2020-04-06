# Reinforcement Learning in CARLA
Code for running Deep Q-Learning in [CARLA](http://carla.org/) simulator.  
CarlaEnv class is similar to the interface used in [OpenAI Gym](https://gym.openai.com/) and can be used with different Reinforcement Learning algorithms.

## Basic setup
1. Create and activate virtual environment in project directory:
```bash
python -m venv venv
source venv/bin/activate
```
2. Install requirements:
```bash
pip install -r requirements.txt 
```
3. Set environment variables in set_environment_variables.sh and source it:
```bash
source ./set_environment_variables.sh
```
4. Start server
```bash
./start_server.sh
```
5. Configure server
```bash
python configure_server.py [map_name]
```
6. Run agent
```bash
python agent.py
```
7. Results are saved in output directory

