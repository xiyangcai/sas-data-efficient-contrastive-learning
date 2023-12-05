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


def synonym_replace(tokens):
    text = ' '.join(tokens)
    aug = naw.SynonymAug(aug_src="wordnet")
    text = aug.augment(text)
    words = word_tokenize(text[0])
    return words
