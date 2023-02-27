#! /bin/bash

pip install -q petals datasets wandb

git clone https://github.com/openai/grade-school-math.git

mkdir gsm8k
mkdir gsm8k/dataset
cp ./grade-school-math/grade_school_math/data/train.jsonl gsm8k/dataset/
cp ./grade-school-math/grade_school_math/data/test.jsonl gsm8k/dataset/

mkdir bloom_weights