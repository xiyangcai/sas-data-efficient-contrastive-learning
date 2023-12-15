# CS260D Project: Apply SAS on Text Data

## Abstract

In this work, we applied the SAS data selection algorithm on the domain of Natural Language Processing (NLP), specifically applied to text classification tasks. We will perform SAS algorithm for data selection and evaluate it on IMDb sentimental analysis task. Further, we also provide certain performance analysis to the experiments.

## Training

To train a proxy model on IMDb dataset:
```shell
python train_text.py
```

Then, you can utilize SAS algorithm for subset selection with the trained proxy model.
```shell
python SubsetSelection.py
```

## Members

Xiyang Cai, Qilin Wang