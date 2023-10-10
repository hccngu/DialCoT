import os
# os.environ["HF_HOME"] = "./hf_home"
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer
import argparse
import pandas as pd
import numpy as np
import json
import re

parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--data', type=str, help='the data used for instructing tuning')
parser.add_argument('--model_type', default="Flan-T5-Base", type=str)  # , choices=['Flan-Alpaca-Base', 'Flan-Alpaca-Large', 'Flan-Alpaca-XL', 'Flan-T5-Base'])
# parser.add_argument('--lora_weights', default="Alpaca-CoT/saved_models/llama-7b-hf_cot", type=str)
parser.add_argument('--test_dataset', default="MultiArith", type=str, choices=['gsm8k', 'SVAMP', 'MultiArith'])
parser.add_argument('--prompt_type', default="Zero-CoT", type=str, choices=['Zero-CoT', 'prompt1', 'prompt2', 'prompt3', 'prompt4'])
parser.add_argument('--device', default=2, type=int)
parser.add_argument('--max_length', default=256, type=int, help='generate max length.')
# parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)
args = parser.parse_args()

if args.model_type == 'Flan-Alpaca-Base':
    model = pipeline(model="declare-lab/flan-alpaca-base", device='cuda:{0}'.format(str(args.device)))
elif args.model_type == 'Flan-Alpaca-Large':
    model = pipeline(model="declare-lab/flan-alpaca-large", device='cuda:{0}'.format(str(args.device)))
elif args.model_type == 'Flan-Alpaca-XL':
    model = pipeline(model="declare-lab/flan-alpaca-xl", device='cuda:{0}'.format(str(args.device)))
elif args.model_type == 'Flan-T5-Base':
    model = pipeline(model="google/flan-t5-base", device='cuda:{0}'.format(str(args.device)))
elif args.model_type == 'Flan-T5-Large':
    model = pipeline(model="google/flan-t5-large", device='cuda:{0}'.format(str(args.device)))
elif args.model_type == 'Flan-T5-XL':
    model = pipeline(model="google/flan-t5-xl", device='cuda:{0}'.format(str(args.device)))
else:
    raise NotImplementedError

if args.prompt_type == 'Zero-CoT':
    prefix_prompt = ""
    one_stage_prompt = " Let's think step by step."
    two_stage_prompt = " Therefore, the answer (arabic numerals) is "
elif args.prompt_type == 'prompt1':
    prefix_prompt = "Q: "
    one_stage_prompt = " Let's give some random thoughts before answering."
    two_stage_prompt = "\nTherefore, the answer is "
elif args.prompt_type == 'prompt2':
    prefix_prompt = "Give stream of consciousness and then the final answer. "
    one_stage_prompt = ""
    two_stage_prompt = "\nThe final answer: "
elif args.prompt_type == 'prompt3':
    prefix_prompt = ""
    one_stage_prompt = " Stream of consciousness first, then make a decision:"
    two_stage_prompt = "\nThus, the answer is "
elif args.prompt_type == 'prompt4':
    prefix_prompt = ""
    one_stage_prompt = " Please break down the question into multiple subquestions."
    two_stage_prompt = " Please answer the above questions in order."
    three_stage_prompt = "\nTherefore, the answer is "
else:
    raise NotImplementedError


if args.test_dataset == 'gsm8k':
    with open('data/gsm8k/test.jsonl', 'r') as f:
        data = f.readlines()
    ans_one_stage = []
    ans_final_stage = []
    target = []
    question = []
    for line in data:
        line_dic = eval(line)
        ins = prefix_prompt + line_dic['question'] + one_stage_prompt
        response = model(ins, max_length=args.max_length, do_sample=True)[0]['generated_text']
        question.append(line_dic['question'])
        print("##Question:", line_dic['question'])
        ans_one_stage.append(response)
        print("##First Stage Response:", response)
        ins = ins + response + two_stage_prompt
        response = model(ins, max_length=args.max_length, do_sample=True)[0]['generated_text']
        ans_final_stage.append(response)
        print("##Second Stage Response:", response)
        target.append(line_dic['answer'])
        print("##Target: ", line_dic['answer'])
        print()
    df = pd.DataFrame({'question': question, 'target': target, 'ans_one_stage': ans_one_stage, 'ans_final_stage': ans_final_stage})

elif args.test_dataset == 'SVAMP':
    with open('data/zero-shot-cot-test-data/SVAMP/SVAMP.json', 'r') as f:
        data = json.load(f)
    if args.prompt_type in ['Zero-CoT', 'prompt1', 'prompt2', 'prompt3']:
        ans_one_stage = []
        ans_final_stage = []
        target_lst = []
        question_lst = []
        # count = 0
        for line in data:
            # count += 1
            # if count == 3:
            #     break
            # line_dic = eval(line)
            question = line['Body'] + line['Question']
            ins = prefix_prompt + question + one_stage_prompt
            response = model(ins, max_length=args.max_length, do_sample=True)[0]['generated_text']
            question_lst.append(question)
            print("##Question:", question)
            ans_one_stage.append(response)
            print("##First Stage Response:", response)
            ins = ins + response + two_stage_prompt
            response = model(ins, max_length=args.max_length, do_sample=True)[0]['generated_text']
            ans_final_stage.append(response)
            print("##Second Stage Response:", response)
            target = line['Equation'] + '####' + str(line['Answer'])
            target_lst.append(target)
            print("##Target: ", target)
            print()
        df = pd.DataFrame({'question': question_lst, 'target': target_lst, 'ans_one_stage': ans_one_stage, 'ans_final_stage': ans_final_stage})
    elif args.prompt_type in ['prompt4']:
        ans_one_stage = []
        ans_two_stage = []
        ans_final_stage = []
        target_lst = []
        question_lst = []
        # count = 0
        for line in data:
            # count += 1
            # if count == 3:
            #     break
            # line_dic = eval(line)
            question = line['Body'] + line['Question']
            ins = prefix_prompt + line['Body'] + line['Question'] + one_stage_prompt
            response = evaluate(ins)
            if response[-4:] == "</s>":
                response = response[:-4]
            question_lst.append(question)
            print("##Question:", question)
            ans_one_stage.append(response)
            print("##First Stage Response:", response)
            ins = line['Body'] + response + line['Question'] + two_stage_prompt
            response = evaluate(ins)
            if response[-4:] == "</s>":
                response = response[:-4]
            ans_two_stage.append(response)
            print("##Second Stage Response:", response)
            ins = line['Body'] + line['Question'] + response + three_stage_prompt
            response = evaluate(ins)
            if response[-4:] == "</s>":
                response = response[:-4]
            ans_final_stage.append(response)
            print("##Final Stage Response:", response)
            target = line['Equation'] + '####' + str(line['Answer'])
            target_lst.append(target)
            print("##Target: ", target)
            print()
        df = pd.DataFrame({'question': question_lst, 'target': target_lst, 'ans_one_stage': ans_one_stage, 'ans_two_stage': ans_two_stage, 'ans_final_stage': ans_final_stage})

elif args.test_dataset == 'MultiArith':
    with open('data/zero-shot-cot-test-data/MultiArith/MultiArith.json', 'r') as f:
        data = json.load(f)
    if args.prompt_type in ['Zero-CoT', 'prompt1', 'prompt2', 'prompt3']:
        ans_one_stage = []
        ans_final_stage = []
        target_lst = []
        question_lst = []
        for line in data:
            # count += 1
            # if count == 3:
            #     break
            # line_dic = eval(line)
            question = line['sQuestion']
            ins = prefix_prompt + question + one_stage_prompt
            response = model(ins, max_length=args.max_length, do_sample=True)[0]['generated_text']
            question_lst.append(question)
            print("##Question:", question)
            ans_one_stage.append(response)
            print("##First Stage Response:", response)
            ins = ins + response + two_stage_prompt
            response = model(ins, max_length=args.max_length, do_sample=True)[0]['generated_text']
            ans_final_stage.append(response)
            print("##Second Stage Response:", response)
            target = line['lEquations'][0] + '####' + str(int(eval(str(line['lSolutions'][0]))))
            target_lst.append(target)
            print("##Target: ", target)
            print()
        df = pd.DataFrame({'question': question_lst, 'target': target_lst, 'ans_one_stage': ans_one_stage, 'ans_final_stage': ans_final_stage})
    elif args.prompt_type in ['prompt4']:
        ans_one_stage = []
        ans_two_stage = []
        ans_final_stage = []
        target_lst = []
        question_lst = []
        # count = 0
        for line in data:
            # count += 1
            # if count == 3:
            #     break
            # line_dic = eval(line)
            question = line['sQuestion']
            ins = prefix_prompt + question + one_stage_prompt
            response = evaluate(ins)
            if response[-4:] == "</s>":
                response = response[:-4]
            question_lst.append(question)
            print("##Question:", question)
            ans_one_stage.append(response)
            print("##First Stage Response:", response)
            ins = response + question + two_stage_prompt
            response = evaluate(ins)
            if response[-4:] == "</s>":
                response = response[:-4]
            ans_two_stage.append(response)
            print("##Second Stage Response:", response)
            ins = question + response + three_stage_prompt
            response = evaluate(ins)
            if response[-4:] == "</s>":
                response = response[:-4]
            ans_final_stage.append(response)
            print("##Final Stage Response:", response)
            target = line['lEquations'][0] + '####' + str(int(eval(str(line['lSolutions'][0]))))
            target_lst.append(target)
            print("##Target: ", target)
            print()
        df = pd.DataFrame({'question': question_lst, 'target': target_lst, 'ans_one_stage': ans_one_stage, 'ans_two_stage': ans_two_stage, 'ans_final_stage': ans_final_stage})
else:
    raise NotImplementedError


count = 0
for i in range(df.shape[0]):
    ans = str(int(eval(re.sub(',', '', df.iloc[i]['target'].split('####')[-1].strip()))))
    try:
        if ans in df.iloc[i]['ans_final_stage']:
            count += 1
    except TypeError:
        continue

acc = count/df.shape[0]
print('correct number:', count)
print('test samples number:', df.shape[0])
print('acc:', acc)

output_file = 'results/test_{0}_{1}_{2}_'.format(args.test_dataset, args.model_type, args.prompt_type) + str(round(acc, 4)*10000) + '.csv'
df.to_csv(output_file, index=False)
# df.to_excel('Alpaca-CoT/results/test_gsm8k_llama_7b.xlsx', index=False)