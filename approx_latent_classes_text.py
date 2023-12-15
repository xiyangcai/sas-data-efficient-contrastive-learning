import os.path
from copy import deepcopy
from typing import List

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fast_pytorch_kmeans import KMeans
from tqdm import tqdm
from torchtext.vocab import GloVe
from transformers import BertTokenizer, BertModel


def glove_approx(
        trainset: torch.utils.data.Dataset,
        labeled_example_indices: List[int],
        labeled_examples_labels: np.array,
        num_classes: int,
        device: torch.device,
        batch_size: int = 512,
        verbose: bool = False,
):
    Z = encode_using_bert(trainset, device)
    clf = train_linear_classifier(
        X=Z[labeled_example_indices],
        y=torch.tensor(labeled_examples_labels),
        representation_dim=len(Z[0]),
        num_classes=num_classes,
        device=device,
        verbose=False
    )
    preds = []
    for start_idx in range(0, len(Z), batch_size):
        preds.append(torch.argmax(clf(Z[start_idx:start_idx + batch_size]).detach(), dim=1).cpu())
    preds = torch.cat(preds).numpy()

    return partition_from_preds(preds)


def encode_using_bert(dataset, device):
    cache_path = 'imdb_trainset_bert_embedding.pt'
    if os.path.exists(cache_path):
        Z = torch.load(cache_path).to(device)
        return Z

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    texts = []

    for i in range(len(dataset)):
        sample = dataset[i]
        text = ' '.join(sample.text)
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded_input)
        last_hidden_states = outputs.last_hidden_state
        sentence_vec = last_hidden_states.mean(dim=1)

        texts.append(sentence_vec)

    Z = torch.cat(texts, dim=0).to(device)
    return Z


def encode_using_glove(dataset, device):
    glove = GloVe(name='6B', dim=100)
    texts = []

    for i in range(len(dataset)):
        sample = dataset[i]
        word_vectors = [glove[word.lower()] for word in sample.text if word.lower() in glove.stoi]
        texts.append(torch.stack(word_vectors).mean(0))

    Z = torch.stack(texts).to(device)
    return Z


def train_linear_classifier(
    X: torch.tensor,
    y: torch.tensor,
    representation_dim: int,
    num_classes: int,
    device: torch.device,
    reg_weight: float = 1e-3,
    n_lbfgs_steps: int = 500,
    verbose=False,
):
    if verbose:
        print('\nL2 Regularization weight: %g' % reg_weight)

    criterion = nn.CrossEntropyLoss()
    X_gpu = X.to(device)
    y_gpu = y.to(device)

    # Should be reset after each epoch for a completely independent evaluation
    clf = nn.Linear(representation_dim, num_classes).to(device)
    clf_optimizer = optim.LBFGS(clf.parameters())
    clf.train()

    for _ in tqdm(range(n_lbfgs_steps), desc="Training linear classifier using fraction of labels", disable=not verbose):
        def closure():
            clf_optimizer.zero_grad()
            raw_scores = clf(X_gpu)
            loss = criterion(raw_scores, y_gpu)
            loss += reg_weight * clf.weight.pow(2).sum()
            loss.backward()
            return loss
        clf_optimizer.step(closure)
    return clf


def partition_from_preds(preds):
    partition = {}
    for i, pred in enumerate(preds):
        if pred not in partition:
            partition[pred] = []
        partition[pred].append(i)
    return partition