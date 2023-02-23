import re
import json
import torch
from tqdm import tqdm, trange
from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM
from random import shuffle
import sys

TASK_COUNT = 100
ITERATIONS_COUNT = 8
PARAMS = {
    "do_sample": True,
    "temperature": 0.7,
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
    prompt = None
    if len(sys.argv) == 3:
        prompt_file_path = sys.argv[1]
        with open(prompt_file_path, 'r') as prompt_file:
            prompt = prompt_file.read()
        results = json.load(open(sys.argv[2], 'r'))
        TASK_COUNT -= len(results["Outputs"])
    elif len(sys.argv) != 1:
        print("Incorrect number of params")
        exit(0)
    test_path = './grade-school-math/grade_school_math/data/test.jsonl'
    input_list, answer_list, cot_prompt = get_input_and_prompt(test_path, prompt)

    MODEL_NAME = "bigscience/bloomz-petals"
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME, request_timeout=300, daemon_startup_timeout=120)
    model = model.cuda()

    for input, answer in tqdm(zip(input_list[:TASK_COUNT], answer_list[:TASK_COUNT])):
        task_data = {
            "Question" : input['question'],
            "Answer": str(answer),
            "BLOOM_answers": []
        }
        for i in trange(ITERATIONS_COUNT):
            tokenized_input = tokenizer(cot_prompt + 'Q: ' + input['question'] + '\nA: ', return_tensors="pt")[
            "input_ids"].cuda()
            output = model.generate(tokenized_input, **PARAMS)
            result = {}
            shift = len(cot_prompt) + len(input['question']) + 3
            task_data["BLOOM_answers"].append(tokenizer.decode(output[0])[shift:])
        results["Outputs"].append(task_data)
        with open('./output_self-consistency_method', 'w') as output_file:
            json.dump(results, output_file)
