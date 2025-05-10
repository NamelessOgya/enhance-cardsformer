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
        policy_model_checkpoint_path
    ):
        self.encoder = load_encoder() #言語情報をembeddingするためのencoder
        self.policy_model = load_policy_model(policy_model_checkpoint_path)

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
    player2_model
):
    game = Hearthstone()
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
    match_num = 100
):
    win_num = 0
    for i in range(match_num/ 2):
        p1_agent = MlAgent(
            "trained_models/prediction_model4715.tar", 
            "trained_policy_model/Cardsformer/Trained_weights_1000000.ckpt"    
        )
        p2_agent = RuleAgent(model_name = "RandomAgent")

        
        win_num += res["Player1"]
    
    for i in range(match_num/ 2):
        p1_agent = RuleAgent(model_name = "RandomAgent")
        p2_agent = MlAgent(
            "trained_models/prediction_model4715.tar", 
            "trained_policy_model/Cardsformer/Trained_weights_1000000.ckpt"    
        )

        
        win_num += res["Player2"]
    
    return win_num/ match_num



if __name__ == "__main__":
    ################################
    # AI同士の対戦
    ################################

    # print("AI vs AI")
    # p1_agent = MlAgent(
    #     "trained_models/prediction_model4715.tar", 
    #     "trained_policy_model/Cardsformer/Trained_weights_1000000.ckpt"    
    # )

    # p2_agent = MlAgent(
    #     "trained_models/prediction_model4715.tar", 
    #     "trained_policy_model/Cardsformer/Trained_weights_1000000.ckpt"
    # )

    # res = battle(
    #     p1_agent,
    #     p2_agent
    # )
    # print(res)

    #########################
    # Rule base agent同士の学習
    #########################


    print("Rule vs Rule")
    win_num = 0
    for i in range(20):
        p1_agent = RuleAgent(model_name = "GreedyAgent")
        p2_agent = RuleAgent(model_name = "RandomAgent")

        res = battle(
            p1_agent,
            p2_agent
        )

        print(res)
        win_num += res["Player1"]

    print(f"win rate {win_num / 20}")
    

    #########################
    # AI vs rule
    #########################

    # print("AI vs Rule")

    # win_num = 0
    # for i in range(20):
    #     p1_agent = MlAgent(
    #         "trained_models/prediction_model4715.tar", 
    #         "trained_policy_model/Cardsformer/Trained_weights_1000000.ckpt"
    #     )
    #     p2_agent = RuleAgent()

    #     res = battle(
    #         p1_agent,
    #         p2_agent
    #     )

    #     print(res)
    #     win_num += res["Player1"]

    # print(f"win rate {win_num / 20}")


########################################
### function for debug
########################################

def dev_gen_asm():
    all_assemblies = list(System.AppDomain.CurrentDomain.GetAssemblies())
    target_asm = None

    print("== print assemblies ==")
    for a in all_assemblies:
        print(a)
        if "SabberStoneAICompetition" in str(a.GetName().Name):
            target_asm = a
            break

    if not target_asm:
        raise Exception("Could not find SabberStoneBasicAI assembly in the current AppDomain.")

    greedy_type = target_asm.GetType("SabberStoneBasicAI.AIAgents.GreedyAgent", throwOnError=True)
    print("Found GreedyAgent type:", greedy_type)

    greedy_agent_obj = System.Activator.CreateInstance(greedy_type)   
    
    getmove_method = greedy_type.GetMethod(
        "GetMove",
        BindingFlags.Instance | BindingFlags.Public,  # クラス自体はinternalだがメソッドはpublic override
        None,  # binder
        [POGame],  # 引数リスト: GreedyAgent.GetMove(POGame)
        None
    )
    print("GreedyAgent.GetMove method =", getmove_method) 

def dev_search_reference():
    import System

    print("=== SabberStoneCore.dll ===")
    asm = clr.AddReference(
        base + "/SabberStoneCore.dll") #修正

    for t in asm.ExportedTypes:
        print(f"{t.Namespace}.{t.Name}")

    print("=== SabberStoneCore.dll ===")
    asm = clr.AddReference(
        base + "/SabberStoneCore.dll") #修正

    for t in asm.ExportedTypes:
        print(f"{t.Namespace}.{t.Name}")

    print("=== SabberStoneAICompetition.dll ===")
    asm = clr.AddReference(
        base + "/SabberStoneAICompetition.dll") #修正
    for t in asm.ExportedTypes:
        print(f"{t.Namespace}.{t.Name}")

