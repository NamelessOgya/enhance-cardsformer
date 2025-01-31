import os
import subprocess

cs_file = "../bot/DrunkenAggroWarriorAgent.cs"

result = subprocess.run(["dotnet", "restore"], capture_output=True, text=True)
result = subprocess.run(["dotnet", "clean", "../HearthstoneAICompetition/core-extensions/SabberStoneBasicAI"], capture_output=True, text=True)

print("Return code:", result.returncode)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

result = subprocess.run(["dotnet", "build", "../HearthstoneAICompetition/core-extensions/SabberStoneBasicAI", "-c", "Release","/p:TreatWarningsAsErrors=false"], capture_output=True, text=True)

print("Return code:", result.returncode)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)