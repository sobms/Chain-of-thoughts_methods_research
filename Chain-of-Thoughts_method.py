import re
import json
import torch
from tqdm import tqdm
from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM
from random import shuffle


TASK_COUNT = 100
PARAMS = {
    "do_sample": None,
    "temperature": 1.0,
    "top_k": None,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 100,
    "num_return_sequences": None,
    "stop": ["Q:"]
}

def make_prompt(cot_prompt, cot_labels):
    prompt = ''
    for example, label in zip(cot_prompt, cot_labels):
        prompt += 'Q: ' + example['question'] + '\nA: ' + example['answer'] + 'The answer is ' + str(label) + '\n\n'
    with open('./prompt_for_CoT_method', 'w') as prompt_file:
        prompt_file.write(prompt)
    return prompt

def get_input_and_prompt(test_path):
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

    cot_prompt = make_prompt(input_list[:11], answer_list[:11])
    answer_list = answer_list[11:]
    input_list = input_list[11:]
    return input_list, answer_list, cot_prompt

if __name__ == '__main__':
    test_path = './grade-school-math/grade_school_math/data/test.jsonl'
    input_list, answer_list, cot_prompt = get_input_and_prompt(test_path)

    MODEL_NAME = "bigscience/bloomz-petals"
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME, request_timeout=300, daemon_startup_timeout=120)
    model = model.cuda()

    results = {
        "Parameters_of_generation": PARAMS,
        "Outputs": []
    }

    for input, answer in tqdm(zip(input_list[:TASK_COUNT], answer_list[:TASK_COUNT])):
        task_data = {
            "Question": input['question'],
            "Answer": str(answer),
            "BLOOM_answer": None
        }
        tokenized_input = tokenizer(cot_prompt + 'Q: ' + input['question'] + '\nA: ', return_tensors="pt")[
            "input_ids"].cuda()
        output = model.generate(tokenized_input, **PARAMS)
        shift = len(cot_prompt) + len(input['question']) + 3
        task_data["BLOOM_answer"] = tokenizer.decode(output[0])[shift:]
        results["Outputs"].append(task_data)
        with open('./output_CoT_method', 'w') as output_file:
            json.dump(results, output_file)
