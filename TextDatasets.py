import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import time
import random

train_loader, valid_loader, test_loader = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True, # necessary for packed_padded_sequence
    device=DEVICE)