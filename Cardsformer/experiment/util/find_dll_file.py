"""
python -m experiment.util.find_dll_file
"""

import os

# 環境変数を設定
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

# 設定した環境変数を取得
print(os.environ["PYTHONNET_RUNTIME"])

import clr
import sys
from System.Reflection import Assembly
from System import Activator

class_name = "LynamicDookaheadAgentV1Master"
base = os.getcwd() + "/../HearthstoneAICompetition/core-extensions/SabberStoneBasicAI/bin/Release/netcoreapp2.1"
clr.AddReference(
    base + "/SabberStoneAICompetition.dll"
) #修正
    
clr.AddReference(
    base + "/SabberStoneCore.dll"
) #修正


def list_up_subclass():
    assembly = Assembly.LoadFile(base + "/SabberStoneAICompetition.dll")

    # 名前空間内のクラスを列挙
    for type in assembly.GetTypes():
        print("Class:", type.FullName)

def get_instance_by_reflection():
    assembly = Assembly.LoadFile(base + "/SabberStoneAICompetition.dll")
    tRandomAgent = assembly.GetType("SabberStoneBasicAI.AIAgents.RandomAgent")
    agent = Activator.CreateInstance(tRandomAgent)
    agent.InitializeAgent()
    agent.InitializeGame()   

    print(t)

if __name__ == "__main__":
    list_up_subclass()
    get_instance_by_reflection()
    
    