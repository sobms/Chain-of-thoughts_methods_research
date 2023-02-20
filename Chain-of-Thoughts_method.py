import re
import json
import torch
from tqdm import tqdm
from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM
from random import shuffle


TASK_COUNT = 100

def make_prompt(cot_prompt, cot_labels):
    prompt = ''
    for example, label in zip(cot_prompt, cot_labels):
        prompt += 'Q: ' + example['question'] + '\nA: ' + example['answer'] + 'The answer is ' + str(label) + '\n\n'
    return prompt

def get_input_and_prompt(test_path):
    with open(test_path, 'r') as data_file:
        test_problems = data_file.readlines()
    # extract final answer
    label_list = [float(re.search(r'#### ([0-9-]+)', problem).group(1)) for problem in test_problems]
    # remove annotations of calculator and final answer, convert string to dict
    input_list = [json.loads(re.sub('<<[0-9\(\)\.=+*/-]*>>|#### ([0-9-]+)', '', problem))
                  for problem in test_problems]
    combined = list(zip(label_list, input_list))
    shuffle(combined)
    label_list[:], input_list[:] = zip(*combined)

    cot_labels = label_list[:11]
    label_list = label_list[11:]
    cot_prompt = make_prompt(input_list[:11], cot_labels)
    input_list = input_list[11:]
    return input_list, label_list, cot_prompt, cot_labels

if __name__ == '__main__':
    test_path = './grade-school-math/grade_school_math/data/test.jsonl'
    input_list, label_list, cot_prompt, cot_labels = get_input_and_prompt(test_path)

    MODEL_NAME = "bigscience/bloomz-petals"
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME, request_timeout=300, daemon_startup_timeout=120)
    model = model.cuda()


    with open('./bloom_output', 'w') as output_file:
        outputs = []
        for input in tqdm(input_list[:TASK_COUNT]):
            tokenized_input = tokenizer(cot_prompt + 'Q: ' + input['question'] + '\nA: ', return_tensors="pt")["input_ids"].cuda()
            output = model.generate(tokenized_input, max_new_tokens=80, stop=['Q:'])
            outputs.append(tokenizer.decode(output[0]))
            json.dump(outputs, output_file)
