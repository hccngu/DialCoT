import shutil
from pathlib import Path
import os
import re
# os.environ["HF_HOME"] = "./hf_home"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pandas as pd
import json
import torch
torch.cuda.set_device(0)

from transformers import T5Tokenizer, T5ForConditionalGeneration, MaxLengthCriteria
from fire import Fire
from huggingface_hub import HfApi
from lightning_fabric import seed_everything

from training import LightningModel
import argparse

def compare_floats(a, b, tolerance=1e-9):
    return abs(a - b) < tolerance

def test_answer(pred_str, ans_str):
    pred_str = str(pred_str)
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if(len(pred) >= 1):
        # print(pred_str)
        pred = float(pred[-1])
        gold = re.findall(pattern, ans_str)
        # print(ans_str)
        gold = float(gold[-1])
        return compare_floats(pred, gold)
    else: 
        return False


def test_model(args, model, tokenizer, query=None):
    if not query:
        raise NotImplementedError

    # model: LightningModel = LightningModel.load_from_checkpoint(path)
    # tokenizer = model.tokenizer
    input_ids = tokenizer(query, return_tensors="pt").input_ids

    if 'hugging' in args.model_type:
        with torch.inference_mode():
            model.eval()
            input_ids = input_ids.to(args.device)
            outputs = model.generate(
                input_ids=input_ids, max_length=args.max_length, do_sample=True
            )
    else:
        seed_everything(model.hparams.seed)
        with torch.inference_mode():
            model.model.eval()
            # model = model.to(args.device)
            input_ids = input_ids.to(args.device)
            outputs = model.model.generate(
                input_ids=input_ids, max_length=args.max_length, do_sample=True
            )

    # print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])

    """
    Example output (outputs/model/base/epoch=2-step=2436.ckpt):
    <pad> Dear [Company Name], I am writing to demonstrate the feasibility of using 42 as an optimal seed
    for training neural networks. I am sure that this seed will be an invaluable asset for the training of 
    these neural networks, so let me know what you think.</s>
    """


def export_checkpoint(path: str, path_out: str):
    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(path_out)
    model.tokenizer.save_pretrained(path_out)


def export_to_hub(path: str, repo: str, temp: str = "temp"):
    if Path(temp).exists():
        shutil.rmtree(temp)

    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(temp)
    model.tokenizer.save_pretrained(temp)
    del model  # Save memory?

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=temp)


"""
huggingface-cli login

p inference.py export_to_hub \
--path "outputs_unclean/model/xl/epoch=2-step=2439.ckpt" \
--repo declare-lab/flan-alpaca-xl

p inference.py export_to_hub \
--path "outputs/model/xxl/epoch=0-step=203.ckpt" \
--repo declare-lab/flan-alpaca-xxl

p inference.py export_to_hub \
--path "outputs/model_gpt4all/xl/epoch=0-step=6838.ckpt" \
--repo declare-lab/flan-gpt4all-xl

p inference.py export_to_hub \
--path "outputs/model_sharegpt/xl/epoch=0-step=4485.ckpt" \
--repo declare-lab/flan-sharegpt-xl

p inference.py export_to_hub \
--path "outputs/model/xl/epoch=2-step=2439.ckpt" \
--repo declare-lab/flan-alpaca-gpt4-xl

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--data', type=str, help='the data used for instructing tuning')
    parser.add_argument('--model_type', default="flan-t5-large", type=str)  # flan-t5-xl-simple-59-ep
    parser.add_argument('--model_path', default="google/flan-t5-large", type=str)
    parser.add_argument('--model_path_decom', default="outputs/flanT5/large/shared_final1_just_gsm8k/100/epoch=93-step=658.ckpt", type=str)
    parser.add_argument('--model_path_solver', default="outputs/flanT5/large/shared_final1_just_gsm8k/100/epoch=93-step=658.ckpt", type=str)
    # parser.add_argument('--lora_weights', default="Alpaca-CoT/saved_models/llama-7b-hf_cot", type=str)
    parser.add_argument('--test_dataset', default="gsm8k", type=str, choices=['gsm8k-200-eval', 'gsm8k', 'SVAMP', 'MultiArith'])
    parser.add_argument('--prompt_type', default="decom2", type=str, choices=['Zero-CoT', 'Few-CoT', 'prompt1', 'prompt2', 'prompt3', 'prompt4', 'decom1', 'decom2', 'decom3', 'selfask-a', 'selfask-s'])
    parser.add_argument('--device', default='cuda:2', type=str)
    parser.add_argument('--max_length', default=512, type=int, help='generate max length.')
    parser.add_argument('--max_response_iter', default=5, type=int, help='max response iter for decom2.')
    # parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)
    args = parser.parse_args()
    print(args)

    args.device = torch.device(args.device)
    if 'hugging' in args.model_type:
        model_decom = T5ForConditionalGeneration.from_pretrained(args.model_path).to(args.device)
        model_solver = T5ForConditionalGeneration.from_pretrained(args.model_path).to(args.device)
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    elif 'xxl' in args.model_type:
        model_decom = model_solver = LightningModel.load_from_checkpoint(args.model_path_decom, map_location=args.device).to(args.device)
        tokenizer = model_decom.tokenizer
    else:
        model_decom = LightningModel.load_from_checkpoint(args.model_path_decom, map_location=args.device).to(args.device)
        model_solver = LightningModel.load_from_checkpoint(args.model_path_solver, map_location=args.device).to(args.device)
        # model_decom = model_decom.to('cuda')
        # model_solver = model_solver.to('cuda')
        tokenizer = model_decom.tokenizer
    

    if args.prompt_type == 'decom1':
        prefix_prompt = 'Content: '
        prompt_decomposer = '\nPlease break down the above question into multiple sub-questions.'
        prompt_solver = 'Please answer the above sub-questions in order and provide the final answer.'
    elif args.prompt_type == 'decom2':
        prefix_decom_prompt = 'Based on the given text, generate the next question that needs to be answered.\nContent: '
        prefix_solver_prompt = 'Based on the given dialogue text, generate a reply.\nContent: '
    elif args.prompt_type == 'decom3':
        prefix_decom_prompt = 'Based on the given text, generate the next question that needs to be answered.\nContent: '
        prefix_solver_prompt = 'Content: '
    elif args.prompt_type == 'Zero-CoT':
        prefix_prompt = ""
        one_stage_prompt = " Let's think step by step."
        two_stage_prompt = " Therefore, the answer (arabic numerals) is "
    elif args.prompt_type == 'Few-CoT':
        prefix_prompt = ""
        # one_stage_prompt = " Let's think step by step."
        # two_stage_prompt = " Therefore, the answer (arabic numerals) is "
    elif args.prompt_type == 'selfask-a':
        prefix_prompt = ""
    elif args.prompt_type == 'selfask-s':
        pass
    else:
        raise NotImplementedError


    if 'gsm8k' in args.test_dataset:
        if args.test_dataset == 'gsm8k':
            with open('data/gsm8k/gsm8k_test.txt', 'r') as f:
                data = f.readlines()
        elif args.test_dataset == 'gsm8k-200-eval':
            with open('gsm8k_validation.jsonl', 'r') as f:
                data = f.readlines()
        else:
            raise NotImplementedError
        if args.prompt_type == 'decom1':
            contents = []
            decomposer = []
            solver = []
            target = []
            for line in data:
                line_dic = eval(line)
                ins = prefix_prompt + line_dic['question'] + prompt_decomposer
                # response = model(ins, max_length=args.max_length, do_sample=True)[0]['generated_text']
                response_decomposer = test_model(args, model_decom, tokenizer, ins)
                contents.append(line_dic['question'])
                print("##Question:", line_dic['question'])
                decomposer.append(response_decomposer)
                print("##Decomposer Response:", response_decomposer)
                ins_solver = ins + '\n' + response_decomposer + prompt_solver
                response_solver = test_model(args, model_solver, tokenizer, ins_solver)
                solver.append(response_solver)
                print("##Solver Response:", response_solver)
                target.append(line_dic['answer'])
                print("##Target: ", line_dic['answer'])
                print()
            df = pd.DataFrame({'contents': contents, 'target': target, 'Decomposer_Response': decomposer, 'Solver_Response': solver})
        elif args.prompt_type == 'decom2':
            contents = []
            decomposer = []
            solver = []
            target = []
            final_solver_response = []
            final_decom_response = []
            generate_final_q_lst = []
            sub_decom_lst = []
            sub_solver_lst = []
            for line in data:
                line_dic = eval(line)
                ins_decom = prefix_decom_prompt + line_dic['question'] + '\nDecomposer: '
                ins_solver = prefix_solver_prompt + line_dic['question']
                response_solver = ''
                generate_final_q = False
                sub_decom = []
                sub_solver = []
                for _ in range(args.max_response_iter):
                    response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins_decom)).strip()
                    sub_decom.append(response_decomposer)
                    ins_solver = ins_solver + response_solver + '\nDecomposer: ' + response_decomposer + '\nSolver: '
                    response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                    sub_solver.append(response_solver)
                    if 'Final question' in response_decomposer:
                        generate_final_q = True
                        break
                    ins_decom = ins_decom + response_decomposer + '\nSolver: ' + response_solver + '\nDecomposer: '
                final_decom_response.append(response_decomposer)
                final_solver_response.append(response_solver)
                generate_final_q_lst.append(generate_final_q)
                contents.append(line_dic['question'])
                decomposer.append(ins_decom)
                solver.append(ins_solver)
                sub_decom_lst.append(sub_decom)
                sub_solver_lst.append(sub_solver)
                target.append(line_dic['answer'])
                print("##Question: ", line_dic['question'])
                print("###Generated Sub Questions: ", sub_decom)
                print("####Generated Sub Answers: ", sub_solver)
                print("#####Generated Final Question: ", response_decomposer)
                print("######Generated Final Answer: ", response_solver)
                print("##Target: ", line_dic['answer'])
                print()
            df = pd.DataFrame({'contents': contents, 'target': target, 'Generated_Sub_Questions': sub_decom_lst, 'Generated_Sub_Answers': sub_solver_lst, 'Generated_Final_Question': final_decom_response, 'Generated_Final_Answer': final_solver_response})
        elif args.prompt_type == 'decom3':
            contents = []
            decomposer = []
            solver = []
            target = []
            final_solver_response = []
            final_decom_response = []
            generate_final_q_lst = []
            sub_decom_lst = []
            sub_solver_lst = []
            for line in data:
                line_dic = eval(line)
                ins_decom = prefix_decom_prompt + line_dic['question'] + '\nDecomposer: '
                content_solver = prefix_solver_prompt + line_dic['question']
                # response_solver = ''
                generate_final_q = False
                sub_decom = []
                sub_solver = []
                for _ in range(args.max_response_iter):
                    response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins_decom)).strip()
                    sub_decom.append(response_decomposer)
                    # ins_solver = ins_solver + response_solver + '\nDecomposer: ' + response_decomposer + '\nSolver: '
                    if ':' in response_decomposer:
                        temp_res = response_decomposer.split(':')[1].strip()
                    else:
                        temp_res = response_decomposer
                    ins_solver = content_solver + '\nQuestion: ' + temp_res
                    response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                    sub_solver.append(response_solver)
                    if 'Final question' in response_decomposer:
                        generate_final_q = True
                        break
                    ins_decom = ins_decom + response_decomposer + '\nSolver: ' + response_solver + '\nDecomposer: '
                    content_solver = content_solver + ' ' + response_solver
                final_decom_response.append(response_decomposer)
                final_solver_response.append(response_solver)
                generate_final_q_lst.append(generate_final_q)
                contents.append(line_dic['question'])
                decomposer.append(ins_decom)
                solver.append(ins_solver)
                sub_decom_lst.append(sub_decom)
                sub_solver_lst.append(sub_solver)
                target.append(line_dic['answer'])
                print("##Question: ", line_dic['question'])
                print("###Generated Sub Questions: ", sub_decom)
                print("####Generated Sub Answers: ", sub_solver)
                print("#####Generated Final Question: ", response_decomposer)
                print("######Generated Final Answer: ", response_solver)
                print("##Target: ", line_dic['answer'])
                print()
            df = pd.DataFrame({'contents': contents, 'target': target, 'Generated_Sub_Questions': sub_decom_lst, 'Generated_Sub_Answers': sub_solver_lst, 'Generated_Final_Question': final_decom_response, 'Generated_Final_Answer': final_solver_response})
        elif args.prompt_type == 'Zero-CoT':
            # with open('data/gsm8k/test.jsonl', 'r') as f:
            #     data = f.readlines()
            ans_one_stage = []
            ans_final_stage = []
            target = []
            question = []
            for line in data:
                line_dic = eval(line)
                ins = prefix_prompt + line_dic['question'] + one_stage_prompt
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                question.append(line_dic['question'])
                print("##Question:", line_dic['question'])
                ans_one_stage.append(response)
                print("##First Stage Response:", response)
                ins = ins + response + two_stage_prompt
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                ans_final_stage.append(response)
                print("##Second Stage Response:", response)
                target.append(line_dic['answer'])
                print("##Target: ", line_dic['answer'])
                print()
            df = pd.DataFrame({'question': question, 'target': target, 'ans_one_stage': ans_one_stage, 'ans_final_stage': ans_final_stage})
        elif args.prompt_type == 'Few-CoT':
            with open('FlanT5-CoT-Specialization/lib_prompt/prompt_original.txt', 'r') as f:
                exemplars = f.read()
            # ans_one_stage = []
            ans_final_stage = []
            target = []
            question = []
            for line in data:
                line_dic = eval(line)
                ins = exemplars + 'Question: ' + line_dic['question'] + "\nLet's think step by step\n"
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                question.append(line_dic['question'])
                print("##Question:", line_dic['question'])
                ans_final_stage.append(response)
                print("##Final Stage Response:", response)
                target.append(line_dic['answer'])
                print("##Target: ", line_dic['answer'])
                print()
            df = pd.DataFrame({'question': question, 'target': target, 'ans_final_stage': ans_final_stage})
        elif args.prompt_type == 'selfask-a':
            contents = []
            solver = []
            target = []
            for line in data:
                line_dic = eval(line)
                ins = prefix_prompt + line_dic['question']
                # response = model(ins, max_length=args.max_length, do_sample=True)[0]['generated_text']
                response = test_model(args, model_decom, tokenizer, ins)
                contents.append(line_dic['question'])
                print("##Question:", line_dic['question'])
                solver.append(response)
                print("##Response:", response)
                target.append(line_dic['answer'])
                print("##Target: ", line_dic['answer'])
                print()
            df = pd.DataFrame({'contents': contents, 'target': target, 'Solver_Response': solver})
        elif args.prompt_type == 'selfask-s':
            contents = []
            decomposer = []
            solver = []
            target = []
            final_solver_response = []
            final_decom_response = []
            generate_final_q_lst = []
            sub_decom_lst = []
            sub_solver_lst = []
            for line in data:
                line_dic = eval(line)
                ins_decom = line_dic['question'] + '\nFollow up: '
                # content_solver = prefix_solver_prompt + line_dic['question']
                # response_solver = ''
                generate_final_q = False
                sub_decom = []
                sub_solver = []
                for _ in range(args.max_response_iter):
                    response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins_decom)).strip()
                    sub_decom.append(response_decomposer)
                    ins_solver = ins_decom + response_decomposer + '\nIntermediate answer: '
                    response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                    sub_solver.append(response_solver)
                    if 'Final question' in response_decomposer:
                        generate_final_q = True
                        break
                    ins_decom = ins_solver + response_solver + '\nFollow up: '
                final_decom_response.append(response_decomposer)
                final_solver_response.append(response_solver)
                generate_final_q_lst.append(generate_final_q)
                contents.append(line_dic['question'])
                decomposer.append(ins_decom)
                solver.append(ins_solver)
                sub_decom_lst.append(sub_decom)
                sub_solver_lst.append(sub_solver)
                target.append(line_dic['answer'])
                print("##Question: ", line_dic['question'])
                print("###Generated Sub Questions: ", sub_decom)
                print("####Generated Sub Answers: ", sub_solver)
                print("#####Generated Final Question: ", response_decomposer)
                print("######Generated Final Answer: ", response_solver)
                print("##Target: ", line_dic['answer'])
                print()
            df = pd.DataFrame({'contents': contents, 'target': target, 'Generated_Sub_Questions': sub_decom_lst, 'Generated_Sub_Answers': sub_solver_lst, 'Generated_Final_Question': final_decom_response, 'Generated_Final_Answer': final_solver_response})
        else:
            raise NotImplementedError
    
    elif args.test_dataset == 'SVAMP':
        with open('data/zero-shot-cot-test-data/SVAMP/SVAMP_test.json', 'r') as f:
            data = json.load(f)
        if args.prompt_type in ['decom1']:
            contents = []
            decomposer = []
            solver = []
            target_lst = []
            for line in data:
                question = line['Body'] + '. ' + line['Question']
                ins = prefix_prompt + question + prompt_decomposer
                response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                contents.append(question)
                print("##Question:", question)
                decomposer.append(response_decomposer)
                print("##Decomposer Response:", response_decomposer)
                ins_solver = ins + '\n' + response_decomposer + prompt_solver
                response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                solver.append(response_solver)
                print("##Solver Response:", response_solver)
                target = line['Equation'] + '####' + str(line['Answer'])
                target_lst.append(target)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'contents': contents, 'target': target_lst, 'Decomposer_Response': decomposer, 'Solver_Response': solver})

        elif args.prompt_type == 'decom2':
            contents = []
            decomposer = []
            solver = []
            target_lst = []
            final_solver_response = []
            final_decom_response = []
            generate_final_q_lst = []
            sub_decom_lst = []
            sub_solver_lst = []
            for line in data:
                question = line['Body'] + '. ' + line['Question']
                target = line['Equation'] + '####' + str(line['Answer'])
                ins_decom = prefix_decom_prompt + question + '\nDecomposer: '
                ins_solver = prefix_solver_prompt + question
                response_solver = ''
                generate_final_q = False
                sub_decom = []
                sub_solver = []
                for _ in range(args.max_response_iter):
                    response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins_decom)).strip()
                    sub_decom.append(response_decomposer)
                    ins_solver = ins_solver + response_solver + '\nDecomposer: ' + response_decomposer + '\nSolver: '
                    response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                    sub_solver.append(response_solver)
                    if 'Final question' in response_decomposer:
                        generate_final_q = True
                        break
                    ins_decom = ins_decom + response_decomposer + '\nSolver: ' + response_solver + '\nDecomposer: '
                final_decom_response.append(response_decomposer)
                final_solver_response.append(response_solver)
                generate_final_q_lst.append(generate_final_q)
                contents.append(question)
                decomposer.append(ins_decom)
                solver.append(ins_solver)
                sub_decom_lst.append(sub_decom)
                sub_solver_lst.append(sub_solver)
                target_lst.append(target)
                print("##Question: ", question)
                print("###Generated Sub Questions: ", sub_decom)
                print("####Generated Sub Answers: ", sub_solver)
                print("#####Generated Final Question: ", response_decomposer)
                print("######Generated Final Answer: ", response_solver)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'contents': contents, 'target': target_lst, 'Generated_Sub_Questions': sub_decom_lst, 'Generated_Sub_Answers': sub_solver_lst, 'Generated_Final_Question': final_decom_response, 'Generated_Final_Answer': final_solver_response})
        
        elif args.prompt_type == 'decom3':
            contents = []
            decomposer = []
            solver = []
            target_lst = []
            final_solver_response = []
            final_decom_response = []
            generate_final_q_lst = []
            sub_decom_lst = []
            sub_solver_lst = []
            for line in data:
                question = line['Body'] + '. ' + line['Question']
                target = line['Equation'] + '####' + str(line['Answer'])
                ins_decom = prefix_decom_prompt + question + '\nDecomposer: '
                content_solver = prefix_solver_prompt + question
                # response_solver = ''
                generate_final_q = False
                sub_decom = []
                sub_solver = []
                for _ in range(args.max_response_iter):
                    response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins_decom)).strip()
                    sub_decom.append(response_decomposer)
                    # ins_solver = ins_solver + response_solver + '\nDecomposer: ' + response_decomposer + '\nSolver: '
                    if ':' in response_decomposer:
                        temp_res = response_decomposer.split(':')[1].strip()
                    else:
                        temp_res = response_decomposer
                    ins_solver = content_solver + '\nQuestion: ' + temp_res
                    response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                    sub_solver.append(response_solver)
                    if 'Final question' in response_decomposer:
                        generate_final_q = True
                        break
                    ins_decom = ins_decom + response_decomposer + '\nSolver: ' + response_solver + '\nDecomposer: '
                    content_solver = content_solver + ' ' + response_solver
                final_decom_response.append(response_decomposer)
                final_solver_response.append(response_solver)
                generate_final_q_lst.append(generate_final_q)
                contents.append(question)
                decomposer.append(ins_decom)
                solver.append(ins_solver)
                sub_decom_lst.append(sub_decom)
                sub_solver_lst.append(sub_solver)
                target_lst.append(target)
                print("##Question: ", question)
                print("###Generated Sub Questions: ", sub_decom)
                print("####Generated Sub Answers: ", sub_solver)
                print("#####Generated Final Question: ", response_decomposer)
                print("######Generated Final Answer: ", response_solver)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'contents': contents, 'target': target_lst, 'Generated_Sub_Questions': sub_decom_lst, 'Generated_Sub_Answers': sub_solver_lst, 'Generated_Final_Question': final_decom_response, 'Generated_Final_Answer': final_solver_response})
        
        elif args.prompt_type == 'Zero-CoT':
            # with open('data/gsm8k/test.jsonl', 'r') as f:
            #     data = f.readlines()
            ans_one_stage = []
            ans_final_stage = []
            target_lst = []
            question_lst = []
            for line in data:
                question = line['Body'] + '. ' + line['Question']
                target = line['Equation'] + '####' + str(line['Answer'])
                ins = prefix_prompt + question + one_stage_prompt
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                question_lst.append(question)
                print("##Question:", question)
                ans_one_stage.append(response)
                print("##First Stage Response:", response)
                ins = ins + response + two_stage_prompt
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                ans_final_stage.append(response)
                print("##Second Stage Response:", response)
                target_lst.append(target)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'question': question_lst, 'target': target_lst, 'ans_one_stage': ans_one_stage, 'ans_final_stage': ans_final_stage})

        elif args.prompt_type == 'Few-CoT':
            with open('FlanT5-CoT-Specialization/lib_prompt/prompt_original.txt', 'r') as f:
                exemplars = f.read()
            # ans_one_stage = []
            ans_final_stage = []
            target_lst = []
            question_lst = []
            for line in data:
                question = line['Body'] + '. ' + line['Question']
                target = line['Equation'] + '####' + str(line['Answer'])
                ins = exemplars + 'Question: ' + question + "\nLet's think step by step\n"
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                question_lst.append(question)
                print("##Question:", question)
                ans_final_stage.append(response)
                print("##Final Stage Response:", response)
                target_lst.append(target)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'question': question_lst, 'target': target_lst, 'ans_final_stage': ans_final_stage})

    elif args.test_dataset == 'MultiArith':
        with open('data/MultiArith/MultiArith_test.json', 'r') as f:
            data = json.load(f)

        if args.prompt_type in ['decom1']:
            contents = []
            decomposer = []
            solver = []
            target_lst = []
            for line in data:
                question = line['sQuestion']
                target = line['lEquations'][0] + '####' + str(int(eval(str(line['lSolutions'][0]))))
                ins = prefix_prompt + question + prompt_decomposer
                response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                contents.append(question)
                print("##Question:", question)
                decomposer.append(response_decomposer)
                print("##Decomposer Response:", response_decomposer)
                ins_solver = ins + '\n' + response_decomposer + prompt_solver
                response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                solver.append(response_solver)
                print("##Solver Response:", response_solver)
                # target = line['Equation'] + '####' + str(line['Answer'])
                target_lst.append(target)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'contents': contents, 'target': target_lst, 'Decomposer_Response': decomposer, 'Solver_Response': solver})
        
        elif args.prompt_type == 'decom2':
            contents = []
            decomposer = []
            solver = []
            target_lst = []
            final_solver_response = []
            final_decom_response = []
            generate_final_q_lst = []
            sub_decom_lst = []
            sub_solver_lst = []
            for line in data:
                question = line['sQuestion']
                target = line['lEquations'][0] + '####' + str(int(eval(str(line['lSolutions'][0]))))
                ins_decom = prefix_decom_prompt + question + '\nDecomposer: '
                ins_solver = prefix_solver_prompt + question
                response_solver = ''
                generate_final_q = False
                sub_decom = []
                sub_solver = []
                for _ in range(args.max_response_iter):
                    response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins_decom)).strip()
                    sub_decom.append(response_decomposer)
                    ins_solver = ins_solver + response_solver + '\nDecomposer: ' + response_decomposer + '\nSolver: '
                    response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                    sub_solver.append(response_solver)
                    if 'Final question' in response_decomposer:
                        generate_final_q = True
                        break
                    ins_decom = ins_decom + response_decomposer + '\nSolver: ' + response_solver + '\nDecomposer: '
                final_decom_response.append(response_decomposer)
                final_solver_response.append(response_solver)
                generate_final_q_lst.append(generate_final_q)
                contents.append(question)
                decomposer.append(ins_decom)
                solver.append(ins_solver)
                sub_decom_lst.append(sub_decom)
                sub_solver_lst.append(sub_solver)
                target_lst.append(target)
                print("##Question: ", question)
                print("###Generated Sub Questions: ", sub_decom)
                print("####Generated Sub Answers: ", sub_solver)
                print("#####Generated Final Question: ", response_decomposer)
                print("######Generated Final Answer: ", response_solver)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'contents': contents, 'target': target_lst, 'Generated_Sub_Questions': sub_decom_lst, 'Generated_Sub_Answers': sub_solver_lst, 'Generated_Final_Question': final_decom_response, 'Generated_Final_Answer': final_solver_response})
        
        elif args.prompt_type == 'decom3':
            contents = []
            decomposer = []
            solver = []
            target_lst = []
            final_solver_response = []
            final_decom_response = []
            generate_final_q_lst = []
            sub_decom_lst = []
            sub_solver_lst = []
            for line in data:
                question = line['sQuestion']
                target = line['lEquations'][0] + '####' + str(int(eval(str(line['lSolutions'][0]))))
                ins_decom = prefix_decom_prompt + question + '\nDecomposer: '
                content_solver = prefix_solver_prompt + question
                # response_solver = ''
                generate_final_q = False
                sub_decom = []
                sub_solver = []
                for _ in range(args.max_response_iter):
                    response_decomposer = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins_decom)).strip()
                    sub_decom.append(response_decomposer)
                    # ins_solver = ins_solver + response_solver + '\nDecomposer: ' + response_decomposer + '\nSolver: '
                    if ':' in response_decomposer:
                        temp_res = response_decomposer.split(':')[1].strip()
                    else:
                        temp_res = response_decomposer
                    ins_solver = content_solver + '\nQuestion: ' + temp_res
                    response_solver = re.sub('<pad>|</s>', '', test_model(args, model_solver, tokenizer, ins_solver)).strip()
                    sub_solver.append(response_solver)
                    if 'Final question' in response_decomposer:
                        generate_final_q = True
                        break
                    ins_decom = ins_decom + response_decomposer + '\nSolver: ' + response_solver + '\nDecomposer: '
                    content_solver = content_solver + ' ' + response_solver
                final_decom_response.append(response_decomposer)
                final_solver_response.append(response_solver)
                generate_final_q_lst.append(generate_final_q)
                contents.append(question)
                decomposer.append(ins_decom)
                solver.append(ins_solver)
                sub_decom_lst.append(sub_decom)
                sub_solver_lst.append(sub_solver)
                target_lst.append(target)
                print("##Question: ", question)
                print("###Generated Sub Questions: ", sub_decom)
                print("####Generated Sub Answers: ", sub_solver)
                print("#####Generated Final Question: ", response_decomposer)
                print("######Generated Final Answer: ", response_solver)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'contents': contents, 'target': target_lst, 'Generated_Sub_Questions': sub_decom_lst, 'Generated_Sub_Answers': sub_solver_lst, 'Generated_Final_Question': final_decom_response, 'Generated_Final_Answer': final_solver_response})
        
        elif args.prompt_type == 'Zero-CoT':
            # with open('data/gsm8k/test.jsonl', 'r') as f:
            #     data = f.readlines()
            ans_one_stage = []
            ans_final_stage = []
            target_lst = []
            question_lst = []
            for line in data:
                question = line['sQuestion']
                target = line['lEquations'][0] + '####' + str(int(eval(str(line['lSolutions'][0]))))
                ins = prefix_prompt + question + one_stage_prompt
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                question_lst.append(question)
                print("##Question:", question)
                ans_one_stage.append(response)
                print("##First Stage Response:", response)
                ins = ins + response + two_stage_prompt
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                ans_final_stage.append(response)
                print("##Second Stage Response:", response)
                target_lst.append(target)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'question': question_lst, 'target': target_lst, 'ans_one_stage': ans_one_stage, 'ans_final_stage': ans_final_stage})

        elif args.prompt_type == 'Few-CoT':
            with open('FlanT5-CoT-Specialization/lib_prompt/prompt_original.txt', 'r') as f:
                exemplars = f.read()
            # ans_one_stage = []
            ans_final_stage = []
            target_lst = []
            question_lst = []
            for line in data:
                question = line['sQuestion']
                target = line['lEquations'][0] + '####' + str(int(eval(str(line['lSolutions'][0]))))
                ins = exemplars + 'Question: ' + question + "\nLet's think step by step\n"
                response = re.sub('<pad>|</s>', '', test_model(args, model_decom, tokenizer, ins)).strip()
                question_lst.append(question)
                print("##Question:", question)
                ans_final_stage.append(response)
                print("##Final Stage Response:", response)
                target_lst.append(target)
                print("##Target: ", target)
                print()
            df = pd.DataFrame({'question': question_lst, 'target': target_lst, 'ans_final_stage': ans_final_stage})

    else:
        raise NotImplementedError
    

if args.prompt_type == 'decom1':
    count = 0
    # accurate_count = 0
    for i in range(df.shape[0]):
        ans = str(int(eval(re.sub(',', '', df.iloc[i]['target'].split('####')[-1].strip()))))
        try:
            # if ans in df.iloc[i]['Solver_Response']:
            if (test_answer(df.iloc[i]['Solver_Response'], ans)):
                count += 1
            # if 'hugging' not in args.model_type:
            #     if 'So the answer is' in df.iloc[i]['Solver_Response']:
            #         if ans in df.iloc[i]['Solver_Response'].split('So the answer is')[-1]:
            #             accurate_count += 1

        except TypeError:
            continue
elif args.prompt_type == 'decom2':
    count = 0
    # accurate_count = 0
    for i in range(df.shape[0]):
        ans = str(int(eval(re.sub(',', '', df.iloc[i]['target'].split('####')[-1].strip()))))
        try:
            # if ans in df.iloc[i]['Generated_Final_Answer']:
            if (test_answer(df.iloc[i]['Generated_Final_Answer'], ans)):
                count += 1
            # if 'So the answer is' in df.iloc[i]['Generated_Final_Answer']:
            #     if ans in df.iloc[i]['Generated_Final_Answer'].split('So the answer is')[-1]:
            #             accurate_count += 1
        except TypeError:
            continue
elif args.prompt_type == 'decom3':
    count = 0
    # accurate_count = 0
    for i in range(df.shape[0]):
        ans = str(int(eval(re.sub(',', '', df.iloc[i]['target'].split('####')[-1].strip()))))
        try:
            if (test_answer(df.iloc[i]['Generated_Final_Answer'], ans)):
            # if ans in df.iloc[i]['Generated_Final_Answer']:
                count += 1
            # if 'So the answer is' in df.iloc[i]['Generated_Final_Answer']:
            #     if ans in df.iloc[i]['Generated_Final_Answer'].split('So the answer is')[-1]:
            #             accurate_count += 1
        except TypeError:
            continue
elif args.prompt_type in ['Zero-CoT', 'Few-CoT']:
    count = 0
    # accurate_count = 0
    for i in range(df.shape[0]):
        ans = str(int(eval(re.sub(',', '', df.iloc[i]['target'].split('####')[-1].strip()))))
        try:
            if (test_answer(df.iloc[i]['ans_final_stage'], ans)):
            # if ans in df.iloc[i]['Generated_Final_Answer']:
                count += 1
            # if 'So the answer is' in df.iloc[i]['Generated_Final_Answer']:
            #     if ans in df.iloc[i]['Generated_Final_Answer'].split('So the answer is')[-1]:
            #             accurate_count += 1
        except TypeError:
            continue
elif args.prompt_type == 'selfask-a':
    count = 0
    for i in range(df.shape[0]):
        ans = str(int(eval(re.sub(',', '', df.iloc[i]['target'].split('####')[-1].strip()))))
        try:
            if (test_answer(df.iloc[i]['Solver_Response'], ans)):
                count += 1
        except TypeError:
            continue
elif args.prompt_type == 'selfask-s':
    count = 0
    for i in range(df.shape[0]):
        ans = str(int(eval(re.sub(',', '', df.iloc[i]['target'].split('####')[-1].strip()))))
        try:
            if (test_answer(df.iloc[i]['Generated_Final_Answer'], ans)):
                count += 1
        except TypeError:
            continue
else:
    raise NotImplementedError

acc = count/df.shape[0]
# accurate_acc = accurate_count/df.shape[0]
print('correct number:', count)
print('test samples number:', df.shape[0])
print('acc:', acc)
# print('accurate_acc:', accurate_acc)

output_file = 'flan-alpaca/results/test_{0}_{1}_{2}_{3}'.format(args.test_dataset, args.model_type, args.prompt_type, str(int(round(acc, 4)*10000))) + '.csv'
df.to_csv(output_file, index=False)