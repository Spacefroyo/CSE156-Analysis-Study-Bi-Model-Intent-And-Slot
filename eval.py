from torch import optim
import numpy as np
import torch

import utils
from utils import get_chunks
from config import device
import config as cfg
from data2index_ver2 import train_data, test_data, index2slot_dict
from model import *

from make_dict import intent_dict, slot_dict

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
        else:
            FP += 1
            now_FP += 1
    for label_chunk in label_chunks:
        if label_chunk not in pred_chunks:
            FN += 1
            now_FN += 1
    if now_FP + now_FN > 0:
        slot_errors[batch_index] = {"intent also wrong": batch_index in intent_errors, "FP":now_FP, "FN":now_FN, "true": label_chunks, "pred": pred_chunks, "sent": sentence_test[0]}

F1_score = 100.0*2*TP/(2*TP+FN+FP)

print('Intent Acc: {:.4f}'.format(100.0*correct_num/total_test))
print('Slot F1 score: {:.4f}'.format(F1_score))


# Error Processing
print('Sentences with UNK:', unks)
print('Total Sentences:', len(test_data))
import json
with open('errors/intent_errors.json', 'w') as f:
    json.dump(intent_errors, f, indent=1)
with open('errors/slot_errors.json', 'w') as f:
    json.dump(slot_errors, f, indent=1)