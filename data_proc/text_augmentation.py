import numpy as np
import random
import nlpaug.augmenter.word as naw
from nltk.tokenize import word_tokenize


def random_deletion(tokens, p=0.2):
    if len(tokens) == 0:
        return tokens

    mask = np.random.rand(len(tokens)) > p
    remaining_tokens = list(np.array(tokens)[mask])

    if len(remaining_tokens) == 0:
        return [random.choice(tokens)]

    return remaining_tokens


def synonym_wordnet_replacement(tokens):
    text = ' '.join(tokens)
    aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.5)
    text = aug.augment(text)
    words = word_tokenize(text[0])
    return words


def synonym_ppdb_replacement(tokens):
    text = ' '.join(tokens)
    aug = naw.SynonymAug(aug_src="ppdb", model_path='./nlpaug_model/ppdb-2.0-tldr', aug_p=0.5)
    text = aug.augment(text)
    words = word_tokenize(text[0])
    return words


def random_window_selection(tokens):
    text_len = len(tokens)
    start, end = 0, 0

    while end - start < text_len / 2:
        rang = np.random.choice(range(text_len), 2)
        start, end = np.sort(rang)
    return tokens[start: end]