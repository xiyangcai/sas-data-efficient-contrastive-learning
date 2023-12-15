#!/bin/bash

probs=(0.1 0.2 0.3 0.4 0.5)

for prob in "${probs[@]}"
do
    python simclr_text.py --aug_type "WordNet" --aug-prob "$prob"
done

echo "All runs completed."

