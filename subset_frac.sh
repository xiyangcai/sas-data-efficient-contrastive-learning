#!/bin/bash


probs=(0.1 0.2 0.3 0.4 0.5)

for prob in "${probs[@]}"
do
    python simclr_text.py --aug-type "WordNet" --aug-prob "$prob"
done

for prob in "${probs[@]}"
do
    python simclr_text.py --aug-type "PPDB" --aug-prob "$prob"
done


# Array of subset_fraction values
subset_fractions=(0.2 0.4 0.6 0.8)

# Loop through each subset_fraction value
for fraction in "${subset_fractions[@]}"
do
    echo "Running subset selection"
    python SubsetSelection.py --subset-fraction "$fraction"
    echo "Running simclr_text.py with random subset with subset_fraction=$fraction"
    python simclr_text.py --random-subset --subset-fraction "$fraction"
    echo "Running simclr_text.py with SAS subset with subset_fraction=$fraction"
    python simclr_text.py --subset-indices "IMDb-$fraction-sas-indices.pkl" --subset-fraction "$fraction"

done

echo "All runs completed."