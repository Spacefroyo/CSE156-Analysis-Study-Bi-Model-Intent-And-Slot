import json
import numpy as np
from make_dict import intent_dict, slot_dict, intent_list, slot_list, words_vocab
from data2index_ver2 import test_data, index2slot_dict
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

intent_errors = json.load(open('errors/intent_errors.json'))
slot_errors = json.load(open('errors/slot_errors.json'))

part = 0

def parse_words(words):
    ret = []
    for word in words:
        if word == 0:
            ret.append('UNK')
        elif word > 1:
            ret.append(words_vocab[word - 2])
    return ret

# Analyze intent detection
if part == 0 or part == 1:
    simul = 0
    unks = 0
    confusion = np.zeros([len(intent_list), len(intent_list)])
    for i in intent_errors:
        error = intent_errors[i]
        true, pred, sent = error["true"], error["pred"], parse_words(error["sent"])
        confusion[true][pred] += 1

        slot_also_error = 1 if i in slot_errors else 1
        simul += slot_also_error

        if 'UNK' in sent:
            unks += 1

        print("True:", intent_list[true], "\t\tPred:", intent_list[pred], "\t\tSentence:", sent)
    print("Simultaneous errors: ", simul)
    print("Unknown errors: ", unks)
    print("Total errors:", len(intent_errors))

    df = pd.DataFrame(confusion, index = intent_list, columns = intent_list)
    sn.heatmap(df, annot=True)
    plt.show()

