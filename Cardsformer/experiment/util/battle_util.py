"""
    エージェント 対 エージェント
    AI 対 AIの対戦を行い、勝率を記録する。

    python -m experiment.util.battle_util
"""

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

class MlAgent:
    """
        obsを受け取り、actionを返す。
    """
    def __init__(
        self, 
        prediction_model_tar_path, 
        policy_model_checkpoint_path
    ):
        self.encoder = load_encoder() #言語情報をembeddingするためのencoder
        self.prediction_model = load_prediction_model(prediction_model_tar_path) 
        self.policy_model = load_policy_model(policy_model_checkpoint_path)

    def action(
        self, 
        obs, 
        options
    ):
        num_options = len(options)

        with torch.no_grad():
            hand_card_embed = self.encoder.encode(obs['hand_card_names']).to(device)
            minion_embed = self.encoder.encode(obs['minion_names']).to(device)
            weapon_embed = self.encoder.encode(obs['weapon_names']).to(device)
            secret_embed = self.encoder.encode(obs['secret_names']).to(device)

            next_state = self.prediction_model(
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

            agent_output = self.policy_model.forward(
                hand_card_embed, 
                minion_embed, 
                secret_embed, 
                weapon_embed, 
                obs, 
                num_options, 
                actor = True
            )

            agent_output = agent_output.argmax()
            action_idx = int(agent_output.cpu().detach().numpy())
            action = options[action_idx]
        
        return action

class RuleAgent:
    def __init__(self, obs):
        pass

    def action(self, obs):
        pass

def battle(
    player1_model,
    player2_model
):
    game = Hearthstone()
    env = Environment(game, device)
    position, obs, options, done, episode_return = env.initial()

    eposode_return = 0
    while True:

        if position == "Player1":
            action = player1_model.action(obs, options)
        elif position == "Player2":
            action = player1_model.action(obs, options)
        else:
            raise ValueError(f"position must be Player1 or Player2 but {position}")
        
        position, obs, options, done, episode_return, next_state = env.step(action, position)
        
        eposode_return += episode_return.item()

        print(episode_return.item())
        
        if done:
            break
            
    return {
        "Player1":1 if eposode_return == 1  else 0,
        "Player2":1 if eposode_return == -1 else 0
    }


if __name__ == "__main__":
    p1_agent = MlAgent(
        "./experiment/prediction_policy_cycle/res/rental_H100_EXP_20250116_0142/prediction_models/cycle_1/prediction_model455.tar", 
        "./experiment/prediction_policy_cycle/res/rental_H100_EXP_20250116_0142/policy_models/cycle_1/Cardsformer/Trained_weights_9990000.ckpt"
    )

    p2_agent = MlAgent(
        "./experiment/prediction_policy_cycle/res/rental_H100_EXP_20250116_0142/prediction_models/cycle_1/prediction_model455.tar", 
        "./experiment/prediction_policy_cycle/res/rental_H100_EXP_20250116_0142/policy_models/cycle_1/Cardsformer/Trained_weights_9990000.ckpt"
    )

    res = battle(
        p1_agent,
        p2_agent
    )

    print(res)