"""
    エージェント 対 エージェント
    AI 対 AIの対戦を行い、勝率を記録する。

    python -m experiment.train_simple_model.util.battle_util
"""

import os

# 環境変数を設定
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

# 設定した環境変数を取得
print(os.environ["PYTHONNET_RUNTIME"])


import clr
base = os.getcwd() + "/../HearthstoneAICompetition/core-extensions/SabberStoneBasicAI/bin/Release/netcoreapp2.1"
clr.AddReference(
    base + "/SabberStoneAICompetition.dll") #修正
    
clr.AddReference(
    base + "/SabberStoneCore.dll") #修正

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
from experiment.train_simple_model.util.model_util import load_policy_model, load_encoder

from SabberStoneBasicAI.CompetitionEvaluation import Agent

# from SabberStoneBasicAI.AIAgents import RandomAgent
from SabberStoneBasicAI.PartialObservation import POGame

import System
from System.Reflection import BindingFlags

import importlib

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_agent_class(agent_name: str):
    """
    SabberStoneBasicAI.AIAgents モジュール内から agent_name に対応するクラスを動的に取得します。
    """
    module = importlib.import_module("SabberStoneBasicAI.AIAgents")
    try:
        agent_class = getattr(module, agent_name)
    except AttributeError:
        raise ValueError(f"Agent {agent_name} は存在しません")
    return agent_class

class MlAgent:
    """
        obsを受け取り、actionを返す。
    """
    def __init__(
        self, 
        policy_model_checkpoint_path,
        device = "cpu", #GPUの場合は数字の文字列
        use_text_feature = True
    ):
        if device == "cpu":
            self.encoder = load_encoder(device = "cpu") #言語情報をembeddingするためのencoder
            self.policy_model = load_policy_model(policy_model_checkpoint_path, device = "cpu", use_text_feature = use_text_feature)
        else:
            self.encoder = load_encoder(device = f"cuda:{device}") #言語情報をembeddingするためのencoder
            self.policy_model = load_policy_model(policy_model_checkpoint_path, device = device, use_text_feature = use_text_feature)

    def action(
        self, 
        obs, 
        env, 
        options
    ):
        num_options = len(options)

        with torch.no_grad():
            hand_card_embed = self.encoder.encode(obs['hand_card_names']).to(device)
            minion_embed = self.encoder.encode(obs['minion_names']).to(device)
            weapon_embed = self.encoder.encode(obs['weapon_names']).to(device)
            secret_embed = self.encoder.encode(obs['secret_names']).to(device)

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
    def __init__(self, model_name):
        AgentClass = get_agent_class(model_name)
        self.agent = AgentClass()

    def action(
        self, 
        obs, 
        env, 
        options
    ):

        # ゲーム環境からpogの取り出し
        pog = POGame(env.Hearthstone.game, False)
        move = self.agent.GetMove(pog)

        return move

def battle(
    player1_model,
    player2_model,
    deck_mode = None
):
    game = Hearthstone(deck_mode=deck_mode)
    env = Environment(game, device)
    position, obs, options, done, episode_return = env.initial()

    eposode_return = 0
    while True:

        if position == "Player1":
            action = player1_model.action(obs ,env , options)
        elif position == "Player2":
            action = player2_model.action(obs ,env , options)
        else:
            raise ValueError(f"position must be Player1 or Player2 but {position}")
        
        position, obs, options, done, episode_return, next_state = env.step(action, position)
        
        eposode_return += episode_return.item()

        
        if done:
            break
            
    return {
        "Player1":1 if eposode_return == 1  else 0,
        "Player2":1 if eposode_return == -1 else 0
    }



def evaluate_model_with_rulebase(
    check_model_dir,
    rule_model_name,
    match_num = 100,
    device = "cpu",
    deck_mode = None,
    use_text_feature = True
):
    win_num = 0

    agent = MlAgent(
        check_model_dir,
        device = device,
        use_text_feature = use_text_feature   
    )
    for i in tqdm(range(int(match_num)), desc = "agent: player1", leave=False, position=0):
        p1_agent = agent
        p2_agent = RuleAgent(model_name = rule_model_name)

        res = battle(
            p1_agent,
            p2_agent,
            deck_mode = deck_mode
        )
        
        win_num += res["Player1"]
    
    for i in tqdm(range(int(match_num)), desc = "agent: player2", leave=False, position=0):
        p1_agent = RuleAgent(model_name = rule_model_name)
        p2_agent = agent

        res = battle(
            p1_agent,
            p2_agent,
            deck_mode = deck_mode
        )
        
        
        win_num += res["Player2"]
    
    return win_num/ (match_num * 2)



if __name__ == "__main__":
    res = evaluate_model_with_rulebase(
        check_model_dir = "./experiment/train_simple_model/res/cf_implement_simple_model_EXP_20250503_2200/policy_models/Cardsformer/Trained_weights_10000.ckpt",
        rule_model_name = "RandomAgent",
        match_num = 50,
        device = "1"
    )

    print(res)
