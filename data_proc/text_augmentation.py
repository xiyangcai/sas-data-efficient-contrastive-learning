import numpy as np
import random
import nlpaug.augmenter.word as naw
from nltk.tokenize import word_tokenize


def random_deletion(tokens, p=0.4):
    if len(tokens) == 0:
        return tokens

    mask = np.random.rand(len(tokens)) > p
    remaining_tokens = list(np.array(tokens)[mask])

    if len(remaining_tokens) == 0:
        return [random.choice(tokens)]

    return remaining_tokens


def synonym_wordnet_replacement(tokens, p=0.3):
    text = ' '.join(tokens)
    aug = naw.SynonymAug(aug_src="wordnet", aug_p=p)
    text = aug.augment(text)
    words = word_tokenize(text[0])
    return words


def synonym_ppdb_replacement(tokens, p=0.3):
    text = ' '.join(tokens)
    aug = naw.SynonymAug(aug_src="ppdb", model_path='./nlpaug_model/ppdb-2.0-tldr', aug_p=p)
    text = aug.augment(text)
    words = word_tokenize(text[0])
    return words


def random_window_selection(tokens, p=0.1):
    text_len = len(tokens)
    cropped_len = int(text_len * p)
    remain_len = text_len - cropped_len

    if cropped_len == 0:
        cropped_len = 1

    start = np.random.randint(cropped_len)
    end = start + remain_len

    return tokens[start: end]
