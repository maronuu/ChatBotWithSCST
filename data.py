import logging
import os
import sys
# import pickle
import cloudpickle
from collections import defaultdict, Counter
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

import cornell_movies_corpus_handler
import utils


### LIMITATION
MAX_LENGTH = 20
MIN_TOKEN_FREQ = 10
### CONSTANTS
UNKNOWN = "#UNK"
BEGIN = "#BEG"
END = "#END"
### RANDOMS
SEED = 1234
### PATHS
EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"

### Logger
log = logging.getLogger("data")


### Related to embedding
def encode_words(word_list, emb_dict):
    idx_list = list()
    idx_list.append(emb_dict[BEGIN])
    for word in word_list:
        word = word.lower()
        idx_list.append(emb_dict.get(word, emb_dict[UNKNOWN]))
    idx_list.append(emb_dict[END])

    return idx_list

def decode_words(idx_list, rev_emb_dict):
    return [rev_emb_dict.get(idx, UNKNOWN) for idx in idx_list]

def encode_phrase_pairs(phrase_pair_list, emb_dict, exclude_unknown=True):
    unknown_token = emb_dict[UNKNOWN]
    encoded_pair_list = list()
    for phrase1, phrase2 in phrase_pair_list:
        enc1 = encode_words(phrase1, emb_dict)
        if exclude_unknown and unknown_token in enc1:
            continue
        enc2 = encode_words(phrase2, emb_dict)
        if exclude_unknown and unknown_token in enc2:
            continue
        encoded_pair_list.append((enc1, enc2))
    
    return encoded_pair_list

def dialogues_to_phrase_pairs(dialogue_list, max_length=None):
    phrase_pair_list = list()
    for dialogue in dialogue_list:
        prev_phrase = None
        for i, cur_phrase in enumerate(dialogue):
            if prev_phrase is None:
                prev_phrase = cur_phrase
                continue
            
            if max_length is None or (len(prev_phrase) <= max_length and len(cur_phrase) <= max_length):
                # if phrase pair meets with constraints, add it.
                phrase_pair_list.append((prev_phrase, cur_phrase))
            prev_phrase = cur_phrase
    
    return phrase_pair_list  

def generate_emb_dict(phrase_pair_list, freq_set):
    emb_dict = dict()
    # constant
    emb_dict[UNKNOWN] = 0
    emb_dict[BEGIN] = 1
    emb_dict[END] = 2
    idx = 3
    for phrase1, phrase2 in phrase_pair_list:
        word_list = [s.lower() for s in chain(phrase1, phrase2)]
        for word in word_list:
            if word not in emb_dict and word in freq_set:
                emb_dict[word] = idx
                idx += 1
    
    return emb_dict

def save_emb_dict(save_dir, emb_dict):
    with open(os.path.join(save_dir, EMB_DICT_NAME), 'wb') as f:
        cloudpickle.dump(emb_dict, f)

def load_emb_dict(load_dir):
    with open(os.path.join(load_dir, EMB_DICT_NAME), 'rb') as f:
        cloudpickle.load(f)


### Related to training data
def batch_iterator(data, batch_size):
    assert isinstance(data, list)
    assert isinstance(batch_size, int)
    i = 0
    while 1:
        batch = data[i * batch_size: (i + 1) * batch_size]
        if len(batch) <= 1:
            break
        yield batch
        i += 1

def ph_ph_pairs_to_ph_phs_pairs(phrase_pairs):
    # phrase_pairs : input data.
    phrase2phrases = defaultdict(list)
    for phrase1, phrase2 in phrase_pairs:
        # cast list -> tuple (list is unhashable)
        tmp = phrase2phrases[tuple(phrase1)]
        tmp.append(phrase2)
    phrase2phrases = list(phrase2phrases.items())

    return phrase2phrases

def train_test_split(data, test_size=None, train_size=None):
    if test_size is None and train_size is None:
        test_size = 0.25
        train_size = 1 - test_size
    elif test_size is None and train_size is not None:
        test_size = 1 - train_size
    elif test_size is not None and train_size is None:
        train_size = 1 - test_size
    else:
        assert (train_size + test_size == 1.)
    
    num_data = len(data)
    num_train = int(num_data * train_size)
    return data[:num_train], data[num_train:]

def pairs_to_packed_seq_without_out(batch, embedding, device):
    batch_size = len(batch)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    questions, responces = zip(*batch)
    # padded matrix of questions
    lengths = [len(q) for q in questions]
    question_mat = np.zeros((batch_size, lengths[0]), dtype=np.int64)
    for i, question in enumerate(questions):
        question_mat[i, :len(question)] = question
    question_tensor = torch.tensor(question_mat).to(device)
    question_seq = rnn_utils.pack_padded_sequence(
        question_tensor, lengths, batch_first=True
    )
    emb = embedding(question_seq.data)
    packed_question_seq = rnn_utils.PackedSequence(
        emb, question_seq.batch_sizes,
    )

    return packed_question_seq, questions, responces

def input_mat_to_packed(input_mat, embedding, device):
    input_tensor = torch.LongTensor([input_mat]).to(device)
    emb = embedding(input_tensor)
    return rnn_utils.pack_padded_sequence(
        emb, [len(input_mat)], batch_first=True,
    )

def pairs_to_packed_seq(batch, embedding, device):
    packed_question_seq, questions, responces = pairs_to_packed_seq_without_out(batch, embedding, device)
    packed_responce_seqs = list()
    for responce in responces:
        # do not use last responce
        # because next answer does not exist.
        resp = input_mat_to_packed(responce[:-1], embedding, device)
        packed_responce_seqs.append(resp)
    
    return packed_question_seq, packed_responce_seqs, questions, responces

# def bleu_score_for_output(output, ref_seq):
#     # output is on gpu.
#     output = torch.max(output.data, dim=1)[1]
#     output = output.cpu().numpy()

#     return utils.calc_bleu_score_for_seq(output, ref_seq)


### Related to loading organized data
def load_dialogue_and_dict(genre_filter, max_length=MAX_LENGTH, min_token_freq=MIN_TOKEN_FREQ):
    # load dialogue list from corpus.
    dialogues = cornell_movies_corpus_handler.load_dialogues(genre_filter=genre_filter)
    if not dialogues:
        log.error("Failed to load dialogues.")
        raise Exception("Failed to load dialogues.")
    log.info(f"{len(dialogues)} dialogues including {sum([len(dialogue) for dialogue in dialogues])} phrases loaded.")
    
    # dialogue list -> phrase pair list (= training data)
    log.info("Converting it into phrase pairs...")
    phrase_pairs = dialogues_to_phrase_pairs(dialogues, max_length=max_length)
    log.info("Successfully converted into phrase pairs.")
    
    # build embedding dict
    log.info("Counting frequency of each word...")
    counter = Counter()
    for dialogue in dialogues:
        for phrase in dialogue:
            counter.update(phrase)
    freqency_set = set(
        [key for key, val in counter.items() if val >= min_token_freq]
    )
    emb_dict = generate_emb_dict(phrase_pairs, freqency_set)
    log.info(f"Dialogues include {len(counter)} unique words in total.| emb_dict_size: {len(emb_dict)}")

    return phrase_pairs, emb_dict
    