import numpy as np
import random


def random_deletion(tokens, p=0.5):
    if len(tokens) == 0:
        return tokens

    mask = np.random.rand(len(tokens)) > p
    remaining_tokens = list(np.array(tokens)[mask])

    if len(remaining_tokens) == 0:
        return [random.choice(tokens)]

    return remaining_tokens
