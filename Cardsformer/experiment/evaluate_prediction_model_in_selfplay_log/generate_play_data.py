import os

# 環境変数を設定
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

# 設定した環境変数を取得
print(os.environ["PYTHONNET_RUNTIME"])

from typing import OrderedDict


# from transformers import AutoModel, AutoTokenizer
import torch

import pandas as pd

import logging

from Env.Hearthstone import Hearthstone
from Env.EnvWrapper import Environment
from Model.PredictionModel import PredictionModel
from Model.ModelWrapper import Model as PolicyModel

from experiment.util.data_util import NpyLogData
from experiment.util.model_util import load_prediction_model, load_policy_model, load_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_selfplay_log(prediction_model_tar_path, policy_model_checkpoint_path):
    prediction_model = load_prediction_model(prediction_model_tar_path)
    encoder = load_encoder()

    game = Hearthstone()
    
    env = Environment(game, device)
    position, obs, options, done, episode_return = env.initial()

    npy_log_data = NpyLogData()
    
    for i in range(1000):
        if i % 100 == 0:
            print(f"match {i} finished")
        elif i < 10:
            print(f"match {i} finished")
        else:
            pass

        while True:
            num_options = len(options)
        
            with torch.no_grad():
                hand_card_embed = encoder.encode(obs['hand_card_names']).to(device)
                minion_embed = encoder.encode(obs['minion_names']).to(device)
                weapon_embed = encoder.encode(obs['weapon_names']).to(device)
                secret_embed = encoder.encode(obs['secret_names']).to(device)
                with torch.no_grad():
                    next_state = prediction_model(
                        [
                            hand_card_embed, 
                            minion_embed, 
                            weapon_embed, 
                            obs['hand_card_scalar_batch'].to(device), 
                            obs['minion_scalar_batch'].to(device), 
                            obs['hero_scalar_batch'].to(device)
                        ]
                    )
                obs['next_minion_scalar'] = next_state[0]
                obs['next_hero_scalar'] = next_state[1]

                model = load_policy_model(policy_model_checkpoint_path)

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
        
            
            npy_log_data.save_data(old_obs, action_idx, next_state, encoder)
            
            if done:
                break


    
    npy_log_data.save_to_npy('off_line_data_vs_policy_model.npy')



if __name__ == "__main__":
    generate_selfplay_log(
        prediction_model_tar_path = "./trained_models/prediction_model4715.tar", 
        policy_model_checkpoint_path = "../../cf_policy_1gpu/Cardsformer/trained_policy_model/Cardsformer/Trained_weights_66000000.ckpt"
    )
