import json
import numpy as np
from make_dict import intent_dict, slot_dict, intent_list, slot_list, words_vocab, slot_by_intent
from data2index_ver2 import test_data, index2slot_dict
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

intent_errors = json.load(open('errors/intent_errors.json'))
slot_errors = json.load(open('errors/slot_errors.json'))

part = 2

# Analyze intent detection
if part == 0 or part == 1:
    simul = 0
    unks = 0
    confusion = np.zeros([len(intent_list), len(intent_list)])
    for i in intent_errors:
        error = intent_errors[i]
        true, pred, sent = error["true"], error["pred"], error["sent"]
        confusion[true][pred] += 1

        slot_also_error = 1 if i in slot_errors else 0
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

#Analyze slot filling
if part == 0 or part == 2:
    slot_types = set()
    for intent in intent_list:
        slots = slot_by_intent[intent.lower()]
        slot_types.update(slots)
    slot_types = list(slot_types)
    slot_types_dict = {}
    for i, slot_type in enumerate(slot_types):
        slot_types_dict[slot_type] = i

    position_only = 0
    simul = 0
    unks = 0
    confusion = np.zeros([8, 8])
    confusion_1_to_1 = np.zeros([len(slot_types), len(slot_types)])
    miss_per_true = {}
    miss_per_pred = {}
    for slot in slot_list:
        if slot.startswith('I-'):
            continue
        elif slot.startswith('B-'):
            slot = slot[2:]
        miss_per_true[slot] = 0
        miss_per_pred[slot] = 0
    for i in slot_errors:
        error = slot_errors[i]
        true, pred, sent = error["true"], error["pred"], error["sent"]
        FP, FN = error["FP"], error["FN"]
        confusion[FP][FN] += 1
        for token in true:
            miss_per_true[token[-1]] += 1
        for token in pred:
            miss_per_pred[token[-1]] += 1

        miss_t, miss_p = None, None
        preds = {tuple(token[:2]): token[-1] for token in pred}
        for j,token in enumerate(true):
            miss_t, token = token[-1], tuple(token[:2])
            if token in preds:
                miss_p = preds[token]
                if miss_t != miss_p:
                    confusion_1_to_1[slot_types_dict[miss_t.lower()]][slot_types_dict[miss_p.lower()]] += 1

        # if FP == 1 and FN == 1:
        #     miss_t, miss_p = None, None
        #     trues = [token[-1] for token in true]
        #     preds = [token[-1] for token in pred]
        #     for j,token in enumerate(trues):
        #         if token != preds[j]:
        #             miss_t = token
        #     for j,token in enumerate(preds):
        #         if token != trues[j]:
        #             miss_p = token
        #     if miss_t == None or miss_p == None:
        #         position_only += 1
        #     else:
        #         confusion_1_to_1[slot_types_dict[miss_t.lower()]][slot_types_dict[miss_p.lower()]] += 1

        intent_also_error = 1 if i in intent_errors else 0
        simul += intent_also_error

        if 'UNK' in sent:
            unks += 1

        # print("True:", slot_list[true], "\t\tPred:", slot_list[pred], "\t\tSentence:", sent)
    print("Simultaneous errors: ", simul)
    print("Unknown errors: ", unks)
    print("Position-only errors: ", position_only)
    print("Total errors:", len(slot_errors))

    # plt.bar(list(miss_per_true.keys()), miss_per_true.values())
    # plt.title("Miss per true")
    # plt.show()

    # plt.bar(list(miss_per_pred.keys()), miss_per_pred.values())
    # plt.title("Miss per pred")
    # plt.show()

    # df = pd.DataFrame(confusion, index = range(len(confusion)), columns = range(len(confusion[0])))
    # sn.heatmap(df, annot=True)
    # plt.show()

    def get_mat_order(mat):
        for _ in range(100):
            loc = np.unravel_index(np.argmax(mat), mat.shape)
            true, pred = loc
            val = mat[loc]
            mat[loc] = 0
            yield true, pred, val

    # loc1 = np.unravel_index(np.argmax(confusion_1_to_1), confusion_1_to_1.shape)
    # true1, pred1 = loc1
    # print("#1 Error: True:", slot_types[true1], "\tPred:", slot_types[pred1], "\tAmt:", confusion_1_to_1[loc1])
    # confusion_1_to_1[loc1] = 0

    # loc2 = np.unravel_index(np.argmax(confusion_1_to_1), confusion_1_to_1.shape)
    # true2, pred2 = loc2
    # print("#2 Error: True:", slot_types[true2], "\tPred:", slot_types[pred2], "\tAmt:", confusion_1_to_1[loc2])

    print("Total 1-1 errors:", np.sum(confusion_1_to_1))
    mat = np.copy(confusion_1_to_1)
    it = 0
    for true, pred, val in get_mat_order(mat):
        if (it > 30):
            break
        if (true == pred):
            continue
        print(f"Error {it}: True:", slot_types[true], "\tPred:", slot_types[pred], "\tAmt:", val)
        it += 1

    df = pd.DataFrame(confusion_1_to_1, index = slot_types, columns = slot_types)
    sn.heatmap(df, annot=True)
    plt.show()