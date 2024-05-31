from io import TextIOWrapper
import urllib.request
import json
import os

dataset_url = "https://raw.githubusercontent.com/sonos/nlu-benchmark/master/2017-06-custom-intent-engines"

vocab_intent_file = 'snips/data/vocab.intent'
vocab_slot_file = 'snips/data/vocab.slot'

atis_vocab = [" ", "'", '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

slot_vocab = set()

def download(prefix, wf: TextIOWrapper):
    temp_file = "temp.json"
    with open(vocab_intent_file) as f:
        for line in f.readlines():
            intent = line.strip()

            # Get train set
            url = f"{dataset_url}/{intent}/{prefix}_{intent}.json"
            print("Retrieving:", url)
            urllib.request.urlretrieve(url, temp_file)
            curfile = open(temp_file)
            dataset = json.load(curfile)
            dataset = dataset[intent]

            for entry in dataset:
                for i, block in enumerate(entry['data']):
                    slot = block['entity'] if 'entity' in block else None

                    # strip leading/trailing " ", lowercase only
                    tokens = ''.join(filter(lambda x : x in atis_vocab, block['text'].lower())).strip().split()

                    # strip "." from end of sentence
                    if i == len(entry['data'])-1 and block['text'].endswith("."):
                        tokens[-1] = tokens[-1][:-1]
                        if (len(tokens[-1]) == 0):
                            tokens.pop()

                    if len(tokens) == 0:
                        continue

                    append = ("O" if not slot else f"B-{slot}")
                    wf.write(f"{tokens[0]}:{append} ")
                    slot_vocab.add(append)

                    append = ("O" if not slot else f"I-{slot}")
                    for token in tokens[1:]:
                        wf.write(f"{token}:{append} ")
                        slot_vocab.add(append)

                wf.write(f"<=> {intent}\n")
    os.remove(temp_file)

download("train", open("snips/data/train_dev", "w"))
download("validate", open("snips/data/test", "w"))

slot_vocab = sorted(list(slot_vocab))
with open(vocab_slot_file, "w") as f:
    for slot in slot_vocab:
        f.write(f"{slot}\n")