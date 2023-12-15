#!/bin/bash

dims=(50 100 200 300)

for dim in "${dims[@]}"
do
    python simclr_text.py --embedding "$dim"
done

dims=(128 256 512)

for dim in "${dims[@]}"
do
    python simclr_text.py --hidden-dim "$dim"
done


echo "All runs completed."

