import time
from pathlib import Path
import pickle


class Logger:
    def __init__(self, sufix=""):
        self.init_time = str(time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.dir_path = "./output/" + self.init_time + sufix
        Path(self.dir_path).mkdir(parents=True, exist_ok=True)

    def save_fit_history(self, loss_hist, acc_hist):
        fit_history_path = self.dir_path + "/fit_history.pickle"
        with open(fit_history_path, "wb") as f:
            data = {"loss": loss_hist, "accuracy": acc_hist}
            pickle.dump(data, f)

    def save_network_details(self, dictionary, model):
        with open(self.dir_path + "/network_details.txt", 'w') as f:
            for key, value in dictionary.items():
                f.write(key + str(value) + "\n")

        model_json = model.to_json()
        with open(self.dir_path + "/model.json", "w") as f:
            f.write(model_json)

    def save_model_weights(self, model, sufix=None):
        filepath = self.dir_path + "/model"
        if sufix is not None:
            filepath += sufix
        model.save_weights(filepath)

    def episode_summary(self, episode, frame, total_reward, epsilon):
        print("episode: " + str(episode) + ", frames: " + str(frame) + 
        ", total_reward: " + str(round(total_reward, 1)) + ", epsilon: " + 
        str(epsilon))
    
    def save_total_rewards(self, total_rewards):
        fit_history_path = self.dir_path + "/total_rewards.pickle"
        with open(fit_history_path, "wb") as f:
            pickle.dump(total_rewards, f)

    def save_epsilons(self, epsilons):
        epsilons_path = self.dir_path + "/epsilons.pickle"
        with open(epsilons_path, "wb") as f:
            pickle.dump(epsilons, f)

    def save_mean_max_q_values(self, mean_max_q_values):
        mean_max_q_values_path = self.dir_path + "/mean_max_q_values.pickle"
        with open(mean_max_q_values_path, "wb") as f:
            pickle.dump(mean_max_q_values, f)

    def save_qtable(self, Q):
        Q_path = self.dir_path + "/Q.pickle"
        with open(Q_path, "wb") as f:
            pickle.dump(Q, f)

    def save_config(self, config):
        config_path = self.dir_path + "/config.ini"
        with open(config_path, 'w') as f:
            config.write(f)
