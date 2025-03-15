import os
import subprocess


if __name__ == '__main__':
    os.system("rm -rf ./HearthstoneAICompetition/core-extensions/SabberStoneBasicAI/obj")
    print("delete success")

    os.system("pwd")

    result = subprocess.run(["dotnet", "restore"], capture_output=True, text=True)
    result = subprocess.run(["dotnet", "clean", "./HearthstoneAICompetition/core-extensions/SabberStoneBasicAI"], capture_output=True, text=True)

    print("Return code:", result.returncode)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    result = subprocess.run(["dotnet", "build", "./HearthstoneAICompetition/core-extensions/SabberStoneBasicAI", "-c", "Release","/p:TreatWarningsAsErrors=false"], capture_output=True, text=True)

    print("Return code:", result.returncode)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)