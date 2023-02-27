from datasets import load_dataset
import re
import os
import json
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch.optim import AdamW
from transformers import BloomTokenizerFast, get_scheduler
from petals import DistributedBloomForCausalLM
from random import shuffle
import sys

TUNING_MODE = 'ptune'
NUM_EPOCHS = 2
NUM_PREFIX_TOKENS = 100
DEVICE = 'cuda'
BATCH_SIZE = 5
LR = 1e-2
WEIGHT_DECAY = 0.0
SEED = 42
MODEL_MAX_LENGTH = 512
MODEL_NAME = "bigscience/bloomz-petals"
PARAMS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_k": None,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 100,
    "num_return_sequences": None,
    "stop": ["Q:"]
}


if __name__ == '__main__':
    prompt_file_path = None
    if len(sys.argv) == 2:
        prompt_file_path = sys.argv[1]
    elif len(sys.argv) != 1:
        print("Incorrect number of params")
        exit(0)
    	
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME,
        pre_seq_len=NUM_PREFIX_TOKENS,
        tuning_mode=TUNING_MODE,
        request_timeout=300,
        daemon_startup_timeout=120
    ).to(DEVICE)

    if prompt_file_path is not None:
        with open(prompt_file_path, 'r') as prompt_file:
            prompt = prompt_file.read()
            cot_prompt = prompt
    else:
        cot_prompt = ''

    def tokenize(input):
        input['question'] = list(map(lambda x: cot_prompt + x, input['question']))
        tokenized_question = tokenizer(input['question'],
                                       padding='max_length',
                                       truncation=True)["input_ids"]
        tokenized_answer = tokenizer(input['answer'],
                                     padding='max_length',
                                     truncation=True)["input_ids"]
        return {'question': tokenized_question,
                'answer': tokenized_answer}

    data_path = './gsm8k/dataset/'
    dataset = load_dataset(data_path)
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"].shuffle(seed=SEED).shard(num_shards=8, index=0)
    valid_dataset = tokenized_dataset["test"].shuffle(seed=SEED).shard(num_shards=4, index=0)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)
    )

    wandb.init(
        project="bloom-gsm_prefix_tuning",
        config={
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_prefix_tokens": NUM_PREFIX_TOKENS,
            "model_name": MODEL_NAME,
            "seed": SEED,
        }
    )

    iter = 0
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_dataloader):
            iter += 1
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            model.train()
            print(batch['question'])
            outputs = model(batch['question'])
            logits = outputs['logits'].permute(0, 2, 1)
            loss = loss_fn(logits, batch['answer'].long())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if iter % 10 == 0:
                save_dir = './bloom_weights/'
                model_save_path = os.path.join(save_dir,
                                               f'model-{iter}.ckpt')
                torch.save(model.state_dict(), model_save_path)
            wandb.log({"Train Loss": loss})
