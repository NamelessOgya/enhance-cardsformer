import os
import re

#### モデルファイル名称をサーチして、最新のモデルを取得する。
def get(): 
    directory_path = "./trained_models"
    max_model = None
    max_value = -1
    for filename in os.listdir(directory_path):
        print(filename)
        try:
            val = int(filename.split("prediction_model")[1].split(".tar")[0])
            print(val)
            if val > max_value:
                max_value = val
                max_model = filename
        except:
            pass
    print(max_model)

if __name__ == "__main__":
    main()
