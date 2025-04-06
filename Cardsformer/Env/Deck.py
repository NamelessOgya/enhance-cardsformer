# python -m Env.Deck 

import yaml
import os
import glob



def load_card_info(split=None):
    # 現在のスクリプトがあるディレクトリ
    base_dir = os.path.dirname(os.path.abspath(__file__)) + "/decklist"
    yaml_files = glob.glob(os.path.join(base_dir, "*.yaml"))

    deck_list = []
    for f in yaml_files:
        
        with open(f, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        
        if split is None:
            deck_list.append(data)
        elif split == "train" and data["split"] == "train":
            deck_list.append(data)
        elif split == "test" and data["split"] == "test":
            deck_list.append(data)
        else:
            pass
    return deck_list

class Deck:
    deck_list = load_card_info(split=None)
    
class TrainDeck:
    deck_list = load_card_info(split="train")

class TestDeck:
    deck_list = load_card_info(split="test")



if __name__ == "__main__":
    
    d = Deck()

    
    print(d.deck_list[0])

    d = Deck()
    print(f"len of all   deck {len(d.deck_list)}")

    d = TrainDeck()
    print(f"len of train deck {len(d.deck_list)}")

    d = TestDeck()
    print(f"len of test  deck {len(d.deck_list)}")
    
    # print(Deck.deck_list[0])



