from torch import optim
import numpy as np
import torch

import utils
from utils import get_chunks
from config import device
import config as cfg
from data2index_ver2 import train_data, test_data, index2slot_dict
from model import *

from make_dict import intent_dict, slot_dict, slot_by_intent, intent_list, slot_list, words_vocab

freq = {}
# test_data = train_data
def parse_words(words):
    ret = []
    for word in words:
        if word == 0:
            ret.append('UNK')
        elif word > 1:
            ret.append(words_vocab[word - 2])
    return ret

intent_model = torch.load('model_intent_best.ckpt')
slot_model = torch.load('model_slot_best.ckpt')

intent_model.eval()
slot_model.eval()

intent_errors = {}
slot_errors = {}

# Evaluation 
total_test = len(test_data)
correct_num = 0
unks = 0
TP, FP, FN = 0, 0, 0
TP_UNK, FP_UNK, FN_UNK = 0, 0, 0
TP_WI, FP_WI, FN_WI = 0, 0, 0
num_tokens, num_tokens_UNK, num_tokens_WI = 0, 0, 0
num_words, num_words_UNK = 0, 0
for batch_index, data_test in enumerate(utils.get_batch(test_data, batch_size=1)):
    sentence_test, real_len_test, slot_label_test, intent_label_test = data_test
    # print(sentence[0].shape, real_len.shape, slot_label.shape)
    x_test = torch.tensor(sentence_test).to(device)

    mask_test = utils.make_mask(real_len_test, batch=1).to(device)
    # Slot model generate hs_test and intent model generate hi_test
    hs_test = slot_model.enc(x_test)
    hi_test = intent_model.enc(x_test)

    # Slot
    slot_logits_test = slot_model.dec(hs_test, hi_test)
    log_slot_logits_test = utils.masked_log_softmax(slot_logits_test, mask_test, dim=-1)
    slot_pred_test = torch.argmax(log_slot_logits_test, dim=-1)
    # Intent
    intent_logits_test = intent_model.dec(hi_test, hs_test, real_len_test)
    log_intent_logits_test = F.log_softmax(intent_logits_test, dim=-1)
    res_test = torch.argmax(log_intent_logits_test, dim=-1)

    if 0 in sentence_test[0]:
        unks += 1
    
    if res_test.item() == intent_label_test[0]:
        correct_num += 1
    else:
        intent_errors[batch_index] = {"true": intent_label_test[0], "pred": res_test.item(), "sent": sentence_test[0]}

    for w in sentence_test[0]:
        num_words_UNK += 1 if w == 0 else 0
        num_words += 1

    # Calc slot F1 score
    
    slot_pred_test = slot_pred_test[0][:real_len_test[0]]
    slot_label_test = slot_label_test[0][:real_len_test[0]]

    slot_pred_test = [int(item) for item in slot_pred_test]
    slot_label_test = [int(item) for item in slot_label_test]

    slot_pred_test = [index2slot_dict[item] for item in slot_pred_test]
    slot_label_test = [index2slot_dict[item] for item in slot_label_test]

    pred_chunks = get_chunks(['O'] + slot_pred_test + ['O'])
    label_chunks = get_chunks(['O'] + slot_label_test + ['O'])

    now_FP, now_FN = 0, 0
    for pred_chunk in pred_chunks:
        if pred_chunk in label_chunks:
            TP += 1
            TP_WI += 1 if pred_chunk[2] not in slot_by_intent[intent_list[intent_label_test[0]].lower()] else 0
            has_UNK = False
            for w in range(pred_chunk[0], pred_chunk[1]+1):
                has_UNK = has_UNK or sentence_test[0][w-1] == 0
            TP_UNK += 1 if has_UNK else 0
        else:
            FP += 1
            now_FP += 1
            FP_WI += 1 if pred_chunk[2] not in slot_by_intent[intent_list[intent_label_test[0]].lower()] else 0
            has_UNK = False
            for w in range(pred_chunk[0], pred_chunk[1]+1):
                has_UNK = has_UNK or sentence_test[0][w-1] == 0
            FP_UNK += 1 if has_UNK else 0
            # if has_UNK:
            #     print(parse_words(sentence_test[0]), label_chunks, pred_chunks)
    for label_chunk in label_chunks:
        if label_chunk[-1] not in freq:
            freq[label_chunk[-1]] = 0
        freq[label_chunk[-1]] += 1
        has_UNK = False
        for w in range(label_chunk[0], label_chunk[1]+1):
            has_UNK = has_UNK or sentence_test[0][w-1] == 0
        num_tokens_UNK += 1 if has_UNK else 0
        num_tokens_WI += 1 if label_chunk[2] not in slot_by_intent[intent_list[intent_label_test[0]].lower()] else 0
        num_tokens += 1
        if label_chunk not in pred_chunks:
            FN += 1
            now_FN += 1
            FN_WI += 1 if label_chunk[2] not in slot_by_intent[intent_list[intent_label_test[0]].lower()] else 0
            has_UNK = False
            for w in range(label_chunk[0], label_chunk[1]+1):
                has_UNK = has_UNK or sentence_test[0][w-1] == 0
            FN_UNK += 1 if has_UNK else 0
            # if has_UNK:
            #     print(parse_words(sentence_test[0]), label_chunks, pred_chunks)

    if now_FP + now_FN > 0:
        slot_errors[batch_index] = {"intent also wrong": batch_index in intent_errors, "FP":now_FP, "FN":now_FN, "true": label_chunks, "pred": pred_chunks, "sent": parse_words(sentence_test[0])}

F1_score = 100.0*2*TP/(2*TP+FN+FP)

print('Intent Acc: {:.4f}'.format(100.0*correct_num/total_test))
print('Slot F1 score: {:.4f}'.format(F1_score))

print(freq)

# Error Processing
print('Sentences with UNK:', unks)
print('Total Sentences:', len(test_data))

print('TP:', TP, '\tTP_UNK:', TP_UNK, '\tTP_WI:', TP_WI)
print('FP:', FP, '\tFP_UNK:', FP_UNK, '\tFP_WI:', FP_WI)
print('FN:', FN, '\tFN_UNK:', FN_UNK, '\tFN_WI:', FN_WI)
print('Tokens:', num_tokens, '\tUNKs:', num_tokens_UNK, '\tWIs:', num_tokens_WI)
print('Words:', num_words, '\tUNKs:', num_words_UNK)

F1_score_known = 100.0*2*(TP-TP_UNK)/(2*(TP-TP_UNK)+(FN-FN_UNK)+(FP-FP_UNK))
F1_score_UNK = 100.0*2*TP_UNK/(2*TP_UNK+FN_UNK+FP_UNK)

F1_score_CI = 100.0*2*(TP-TP_WI)/(2*(TP-TP_WI)+(FN-FN_WI)+(FP-FP_WI))
F1_score_WI = 100.0*2*TP_WI/(2*TP_WI+FN_WI+FP_WI)
print('Slot F1 score (known): {:.4f}'.format(F1_score_known))
print('Slot F1 score (UNK): {:.4f}'.format(F1_score_UNK))
print('Slot F1 score (CI): {:.4f}'.format(F1_score_CI))
print('Slot F1 score (WI): {:.4f}'.format(F1_score_WI))
import json
with open('errors/intent_errors.json', 'w') as f:
    json.dump(intent_errors, f, indent=1)
with open('errors/slot_errors.json', 'w') as f:
    json.dump(slot_errors, f, indent=1)