import config as cfg

def convert_int(arr):
    try:
        a = int(arr)
    except:
        return None
    return a

# Make words dict	
words = []
with open(cfg.train_file) as f:
    for line in f.readlines():
        line = line.strip().lower().split()

        for index, item in enumerate(line):
            word = item.split(':')[0]
            if word == '<=>':
                break
            if convert_int(word) is not None:
                words.append('DIGIT' * len(word))
            else:        
                words.append(word)

words_vocab = sorted(set(words))
word_dict = {'UNK': 0, 'PAD': 1}

for i, item in enumerate(words_vocab):
    word_dict[item] = i + 2

# Make slot tag dict 
slot_dict = {}
slot_list = []

with open(cfg.vocab_slot_file) as f:

    for i, line in enumerate(f.readlines()):
        slot_dict[line.strip()] = i
        slot_list.append(line.strip())


# print(slot_dict)

# Make intent dict 
intent_dict = {}
intent_list = []

with open(cfg.vocab_intent_file) as f:
    for i, line in enumerate(f.readlines()):
        intent_dict[line.strip()] = i
        intent_list.append(line.strip())

# print(intent_dict)


# Slot by intent
slot_by_intent = {}
for intent in intent_list:
    slot_by_intent[intent.lower()] = set()
with open(cfg.train_file) as f:
    for line in f.readlines():
        line = line.strip().lower().split()

        for index, item in enumerate(line):
            if item == '<=>':
                break
            label = item.split(':')[1][2:]
            if label == 'O':
                continue
            slot_by_intent[line[-1]].add(label)
for intent in intent_list:
    slot_by_intent[intent.lower()] = list(slot_by_intent[intent.lower()])
import json
with open('snips/data/slot_by_intent.json', 'w') as f:
    json.dump(slot_by_intent, f, indent=1)