

import yaml
import os
import glob



def load_card_info():
    # 現在のスクリプトがあるディレクトリ
    base_dir = os.path.dirname(os.path.abspath(__file__)) + "/decklist"
    yaml_files = glob.glob(os.path.join(base_dir, "*.yaml"))

    deck_list = []
    for f in yaml_files:
        
        with open(f, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        
        deck_list.append(data)
    return deck_list

class Deck:
    deck_list = load_card_info()
    




if __name__ == "__main__":
    
    d = Deck()

    
    print(d.deck_list[0])
    
    # print(Deck.deck_list[0])



