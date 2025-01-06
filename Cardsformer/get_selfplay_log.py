import os

# 環境変数を設定
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

# 設定した環境変数を取得
print(os.environ["PYTHONNET_RUNTIME"])

from typing import OrderedDict
from Env.Hearthstone import Hearthstone
from Env.EnvWrapper import Environment
from Model.PredictionModel import PredictionModel
from Model.ModelWrapper import Model as PolicyModel

from transformers import AutoModel, AutoTokenizer
import torch
from Algo.encoder import Encoder
import pandas as pd

import logging
import numpy as np

class NpyLogData:
    def __init__(self):
        self.data = {
            'hand_card_names': [],
            'minion_names': [],
            'weapon_names': [],
            'hand_card_scalar': [],
            'minion_scalar': [],
            'hero_scalar': [],
            'next_state_minion_scalar': [],
            'next_state_hero_scalar': [],
        }

    def save_data(self, old_obs, action_idx,next_state):
        self.data['hand_card_names'].append(encoder.encode(old_obs['hand_card_names']).detach().cpu().numpy())
        self.data['minion_names'].append(encoder.encode(old_obs['minion_names']).detach().cpu().numpy())
        self.data['weapon_names'].append(encoder.encode(old_obs['weapon_names']).detach().cpu().numpy())
        self.data['hand_card_scalar'].append(old_obs['hand_card_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
        self.data['minion_scalar'].append(old_obs['minion_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
        self.data['hero_scalar'].append(old_obs['hero_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
        self.data['next_state_minion_scalar'].append(next_state['minion_scalar'].detach().cpu().numpy().astype(np.int64))
        self.data['next_state_hero_scalar'].append(next_state['hero_scalar'].detach().cpu().numpy().astype(np.int64))

    def save_to_npy(self, npy_name):
        np.save('off_line_data_vs_ai' + str(j) + '.npy', self.data)
    
if __name__ == "__main__":
    shandle = logging.StreamHandler()
    shandle.setFormatter(
        logging.Formatter(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
            '%(message)s'))
    log = logging.getLogger('Cardsformer')
    log.propagate = False
    log.addHandler(shandle)
    log.setLevel(logging.INFO)


    name_list = pd.read_csv('Env/classical_cards.csv')['name'].tolist()

    NUM_ROUNDS = 1
    checkpoint_path = "./trained_policy_model/Cardsformer/Trained_weights_1000000.ckpt"
    device_number = 'cpu'


    model =PolicyModel(device=device_number)
    checkpoint_states = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    model.get_model().load_state_dict(checkpoint_states)

    prediction_model = PredictionModel()
    # prediction
    checkpoint_states = torch.load("./trained_models/prediction_model4715.tar", map_location="cpu")['model_state_dict']

    # unwrap the prediction model
    new_state_dict = OrderedDict()
    for k, v in checkpoint_states.items():
        name = k[7:]
        new_state_dict[name] = v

    prediction_model.load_state_dict(new_state_dict)
    prediction_model.to("cpu")
    prediction_model.eval()
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    encoder = Encoder(model=auto_model, tokenizer=tokenizer)
    encoder.to(device)


    win = [0, 0]

    npy_log_data = NpyLogData()

    for j in range(1):
        game = Hearthstone()
        env = Environment(game, device)

        position, obs, options, done, episode_return = env.initial()
        for i in range(1000):
            if i % 100 == 0:
                print(f"match {i} finished")
            
            while True:
                
                num_options = len(options)
            
                hand_card_embed = encoder.encode(obs['hand_card_names'])
                minion_embed = encoder.encode(obs['minion_names'])
                weapon_embed = encoder.encode(obs['weapon_names'])
                secret_embed = encoder.encode(obs['secret_names'])
                with torch.no_grad():
            
                    hand_card_embed = encoder.encode(obs['hand_card_names'])
                    minion_embed = encoder.encode(obs['minion_names'])
                    weapon_embed = encoder.encode(obs['weapon_names'])
                    secret_embed = encoder.encode(obs['secret_names'])
                    with torch.no_grad():
                        next_state = prediction_model([hand_card_embed, minion_embed, weapon_embed, obs['hand_card_scalar_batch'], obs['minion_scalar_batch'], obs['hero_scalar_batch']])
                    obs['next_minion_scalar'] = next_state[0]
                    obs['next_hero_scalar'] = next_state[1]
                    with torch.no_grad():
                        agent_output = model.forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, actor = True)
                    # uncomment the following line to see the action-value results of each available action
                    # for i in range(num_options):
                    #     log.info('--ACTION-- {} --VALUE-- {}'.format(options[i].FullPrint(), agent_output.reshape(-1)[i]))
                    agent_output = agent_output.argmax()
                    action_idx = int(agent_output.cpu().detach().numpy())
                action = options[action_idx]
            
                player = position
                old_obs = obs
                # log.info(action.FullPrint())  # print the performing action
                position, obs, options, done, episode_return, next_state = env.step(action, player)
                # log.info(env.Hearthstone.game.FullPrint())  # print current game state
            
                
                npy_log_data.save_data(old_obs, action_idx, next_state)
                
                if done:
                    if episode_return > 0:
                        win[0] += 1
                    elif episode_return < 0:
                        win[1] += 1
                    else:
                        log.info("No winner???")
                    break


            
        npy_log_data.save_to_npy('off_line_data_vs_policy_model' + str(j) + '.npy')