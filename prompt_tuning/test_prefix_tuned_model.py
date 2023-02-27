#to run use "python3   test_prefix_tuned_model.py <path_to_model> <prompt_file_path> [<results_file>]

from datasets import load_dataset
import re
import json
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch.optim import AdamW
from transformers import BloomTokenizerFast, get_scheduler
from petals import DistributedBloomForCausalLM
from random import shuffle
import sys

DEVICE = 'cuda'
NUM_PREFIX_TOKENS = 100
MODEL_MAX_LENGTH = 1024
MODEL_NAME = "bigscience/bloomz-petals"
TUNING_MODE = 'ptune'
PARAMS = {
    "do_sample": None,
    "temperature": 1,
    "top_k": None,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 100,
    "num_return_sequences": None,
    "stop": ["Q:"]
}

def make_prompt(cot_prompt, cot_answers):
    prompt = ''
    for example, answer in zip(cot_prompt, cot_answers):
        prompt += 'Q: ' + example['question'] + '\nA: ' + example['answer'] + 'The answer is ' + str(answer) + '\n\n'
    with open('./prompt_for_self-consistency_method', 'w') as prompt_file:
        prompt_file.write(prompt)
    return prompt

def get_input_and_prompt(test_path, prompt):
    with open(test_path, 'r') as data_file:
        test_problems = data_file.readlines()
    # extract final answer
    answer_list = [float(re.search(r'#### ([0-9-]+)', problem).group(1)) for problem in test_problems]
    # remove annotations of calculator and final answer, convert string to dict
    input_list = [json.loads(re.sub('<<[0-9\(\)\.=+*/-]*>>|#### ([0-9-]+)', '', problem))
                  for problem in test_problems]
    combined = list(zip(answer_list, input_list))
    shuffle(combined)
    answer_list[:], input_list[:] = zip(*combined)
    if prompt is not None:
        cot_prompt = prompt
    else:
        cot_prompt = make_prompt(input_list[:11], answer_list[:11])
        answer_list = answer_list[11:]
        input_list = input_list[11:]
    return input_list, answer_list, cot_prompt

if __name__ == '__main__':
    results = {
        "Parameters_of_generation": PARAMS,
        "Outputs": []
    }
    prompt_file_path = None
    path_to_model = None
    if len(sys.argv) == 4:
        results = json.load(open(sys.argv[3], 'r'))
    elif len(sys.argv) != 3:
        print("Incorrect number of params")
        exit(0)
    path_to_model = sys.argv[1]
    prompt_file_path = sys.argv[2]
    with open(prompt_file_path, 'r') as prompt_file:
        prompt = prompt_file.read()
	
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME,
        pre_seq_len=NUM_PREFIX_TOKENS, 
        request_timeout=300, 
        tuning_mode=TUNING_MODE,
        daemon_startup_timeout=120
    ).to(DEVICE)
	
    model.load_state_dict(torch.load(path_to_model))
	
    with open(prompt_file_path, 'r') as prompt_file:
        prompt = prompt_file.read()
        start_idx = len(results["Outputs"])
    test_path = './grade-school-math/grade_school_math/data/test.jsonl'
    input_list, answer_list, cot_prompt = get_input_and_prompt(test_path, prompt)
    TASK_COUNT = len(input_list)
    with model.inference_session(max_length=1024) as sess:
        for input, answer in tqdm(zip(input_list[start_idx:TASK_COUNT], answer_list[start_idx:TASK_COUNT])):
            task_data = {
                "Question" : input['question'],
	        "Answer": str(answer),
	        "BLOOM_answer": None
	    }
	    
            tokenized_input = tokenizer(cot_prompt + 'Q: ' + input['question'] + '\nA: ', return_tensors="pt")["input_ids"].cuda()
            output = model.generate(tokenized_input, **PARAMS, session=sess)
            shift = len(cot_prompt) + len(input['question']) + 3
            task_data["BLOOM_answer"] = tokenizer.decode(output[0])[shift:]
            results["Outputs"].append(task_data)
            with open('./output_CoT_method_prefix_tuned', 'w') as output_file:
                json.dump(results, output_file)
