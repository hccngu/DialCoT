import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_LAUNCH_BLOCKING"]  = '1'
os.environ["HF_HOME"] = "./hf_home"
import re
import gym
from gym import spaces
import numpy as np
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, MaxLengthCriteria
from huggingface_hub import HfApi
from lightning_fabric import seed_everything

import pytorch_lightning as pl
import torch
# from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning import seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.strategies import FSDPStrategy
# from torch.distributed.fsdp import (
#     MixedPrecision,
#     FullyShardedDataParallel,
#     StateDictType,
#     FullStateDictConfig,
# )
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor
# from transformers.models.t5.modeling_t5 import T5Block

# from training import LightningModel


def compare_floats(a, b, tolerance=1e-9):
    return abs(a - b) < tolerance


def test_answer(pred_str, ans_str):
    try: 
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
    except:
        return False


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        print(self.hparams)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name_or_path
        )
        print(dict(orig_state_dict=len(self.model.state_dict())))
        if self.hparams.use_lora:
            raise NotImplementedError
        if self.hparams.use_compile:
            self.model = torch.compile(self.model)
        if self.hparams.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = self.trainer.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # noinspection PyTypeChecker
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            relative_step=False,
        )
        return [optimizer]

    # def train_dataloader(self):
    #     dataset = TextToTextDataset(
    #         path=self.hparams.data_path,
    #         max_source_length=self.hparams.max_source_length,
    #         max_target_length=self.hparams.max_target_length,
    #         tokenizer=self.tokenizer,
    #     )

    #     return DataLoader(
    #         dataset,
    #         batch_size=self.hparams.train_batch_size,
    #         drop_last=True,
    #         shuffle=True,
    #     )


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
                input_ids=input_ids, max_length=args.max_length, do_sample=False, num_beams=args.n_actions, num_return_sequences=args.n_actions, return_dict_in_generate=True, temperature=args.T
            )
    else:
        seed_everything(model.hparams.seed)
        with torch.inference_mode():
            model.model.eval()
            # model = model.to(args.device)
            input_ids = input_ids.to(args.device)
            outputs = model.model.generate(
                input_ids=input_ids, max_length=args.max_length, do_sample=False, num_beams=args.n_actions, num_return_sequences=args.n_actions, return_dict_in_generate=True, temperature=args.T
            )

            temp_inputs = torch.cat((input_ids.repeat(args.n_actions, 1), outputs[0]), dim=1)
            last_hidden_states = model.model.encoder(temp_inputs.clone()).last_hidden_state[:, -1, :]  # [3, hidden_size]
            answer = [tokenizer.decode(o) for o in outputs[0]]
            del input_ids, outputs
            torch.cuda.empty_cache()
        # print(answer)

    # print(tokenizer.decode(outputs[0]))
    return temp_inputs, last_hidden_states.flatten().reshape(1,-1), answer

class OurEnv(gym.Env):

    def __init__(self, args):
        super(OurEnv, self).__init__()

        self.args = args
        self._max_episode_steps = 1000-1
        # 1. 实例化已经train好的模型，加载对应的tokenizer
        args.device = torch.device(args.device)
        if 'hugging' in args.model_type:
            self.model_decom = T5ForConditionalGeneration.from_pretrained(args.model_path).to(args.device)
            self.model_solver = T5ForConditionalGeneration.from_pretrained(args.model_path).to(args.device)
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        elif 'shared' in args.model_type:
            self.model_decom = LightningModel.load_from_checkpoint(args.model_path_decom, map_location=args.device).to(args.device)
            self.tokenizer = self.model_decom.tokenizer
        else:
            self.model_decom = LightningModel.load_from_checkpoint(args.model_path_decom, map_location=args.device).to(args.device)
            self.model_solver = LightningModel.load_from_checkpoint(args.model_path_solver, map_location=args.device).to(args.device)
            # model_decom = model_decom.to('cuda')
            # model_solver = model_solver.to('cuda')
            self.tokenizer = self.model_decom.tokenizer

        # prompts
        if args.prompt_type == 'decom1':
            self.prefix_prompt = 'Content: '
            self.prompt_decomposer = '\nPlease break down the above question into multiple sub-questions.'
            self.prompt_solver = 'Please answer the above sub-questions in order and provide the final answer.'
        elif args.prompt_type == 'decom2':
            self.prefix_decom_prompt = 'Based on the given text, generate the next question that needs to be answered.\nContent: '
            self.prefix_solver_prompt = 'Based on the given dialogue text, generate a reply.\nContent: '
        elif args.prompt_type == 'decom3':
            self.prefix_decom_prompt = 'Based on the given text, generate the next question that needs to be answered.\nContent: '
            self.prefix_solver_prompt = 'Content: '
        elif args.prompt_type == 'Zero-CoT':
            self.prefix_prompt = ""
            self.one_stage_prompt = " Let's think step by step."
            self.two_stage_prompt = " Therefore, the answer (arabic numerals) is "
        elif args.prompt_type == 'Few-CoT':
            self.prefix_prompt = ""
            # one_stage_prompt = " Let's think step by step."
            # two_stage_prompt = " Therefore, the answer (arabic numerals) is "
        else:
            raise NotImplementedError

        # 2. 读入所有的训练数据的source，shuffle
        if args.train_dataset == 'gsm8k':
            # with open('data/gsm8k/train_socratic.jsonl', 'r') as f:
            with open('data/gsm8k/gsm8k_dev.txt', 'r') as f:
                self.data = f.readlines()
            self.data = [d.strip() for d in self.data]
        # elif args.test_dataset == 'gsm8k-200-eval':
        #     with open('data/gsm8k/gsm8k_validation.jsonl', 'r') as f:
        #         data = f.readlines()
        elif args.train_dataset == 'temp':
            with open('data/gsm8k/temp.txt', 'r') as f:
                self.data = f.readlines()
            self.data = [d.strip() for d in self.data]
        elif args.train_dataset == 'temp_5':
            with open('data/gsm8k/gsm8k_test.txt', 'r') as f:
                self.data = f.readlines()
            self.data = [d.strip() for d in self.data]
        else:
            raise NotImplementedError
        
        # self.Q = Q
        # self.R = R
        # self.state = np.array([0.2, 0.7])
        self.action_space = spaces.Discrete(args.n_actions)
        if 'large' in args.model_type:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(args.n_actions*1024, ), dtype=np.float64)
        elif 'xl' in args.model_type:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(args.n_actions*2048, ), dtype=np.float64)
        elif 'xxl' in args.model_type:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(args.n_actions*4096, ), dtype=np.float64)
        elif 'base' in args.model_type:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(args.n_actions*512, ), dtype=np.float64)
        else:
            raise NotImplementedError
        
    def reset(self, data_index):
        line_dic = eval(self.data[data_index])
        self.question = line_dic['question']
        self.target = line_dic['answer']
        self.processed_target = re.split('\n', line_dic['answer'])[:-1]
        self.ins_decom = self.prefix_decom_prompt + line_dic['question'] + '\nDecomposer: '
        self.ins_solver = self.prefix_solver_prompt + line_dic['question']
        previous_string_idx, state, answer = test_model(self.args, self.model_decom, self.tokenizer, self.ins_decom)
        self.answer = answer
        return previous_string_idx, state, answer
        # print(response_decomposer)
        # raise NotImplementedError
    
    def only_eval_reset(self, data):
        self.question = data['question']
        self.target = data['answer']
        self.processed_target = re.split('\n', data['answer'])[:-1]
        self.ins_decom = self.prefix_decom_prompt + data['question'] + '\nDecomposer: '
        self.ins_solver = self.prefix_solver_prompt + data['question']
        previous_string_idx, state, answer = test_model(self.args, self.model_decom, self.tokenizer, self.ins_decom)
        self.answer = answer
        return previous_string_idx, state, answer

    def step(self, action, previous_string_idx, episode_steps):
        done = False
        reward = 0
        if episode_steps % 2 == 1:
            self.ins_solver = self.ins_solver + '\nDecomposer: ' + self.answer[action[0]] + '\nSolver: '
            self.ins_decom = self.ins_decom + self.answer[action[0]]
            previous_string_idx, state, self.answer = test_model(self.args, self.model_decom, self.tokenizer, self.ins_solver)
        else:
            self.ins_decom =  self.ins_decom + '\nSolver: ' + self.answer[action[0]] + '\nDecomposer: '
            self.ins_solver = self.ins_solver + self.answer[action[0]]
            if 'Final question' in self.ins_decom or episode_steps == 10:
                done = True
            if done:
                if (test_answer(self.ins_decom, self.target)):
                    reward = 1
            else:
                if (episode_steps//2)-1 < len(self.processed_target):
                    if (test_answer(self.ins_decom, self.processed_target[(episode_steps//2)-1])):
                        reward = 0.3
            previous_string_idx, state, self.answer = test_model(self.args, self.model_decom, self.tokenizer, self.ins_decom)
        
        return state, reward, done, previous_string_idx, self.answer
    
    def only_eval_step(self, action, previous_string_idx, episode_steps):
        flag = False
        done = False
        reward = 0
        if episode_steps % 2 == 1:
            self.ins_solver = self.ins_solver + '\nDecomposer: ' + self.answer[action[0]] + '\nSolver: '
            self.ins_decom = self.ins_decom + self.answer[action[0]]
            previous_string_idx, state, self.answer = test_model(self.args, self.model_decom, self.tokenizer, self.ins_solver)
        else:
            self.ins_decom =  self.ins_decom + '\nSolver: ' + self.answer[action[0]] + '\nDecomposer: '
            self.ins_solver = self.ins_solver + self.answer[action[0]]
            if 'Final question' in self.ins_decom or episode_steps == 10:
                done = True
            if done:
                if (test_answer(self.ins_decom, self.target)):
                    reward = 1
                    flag = True
            else:
                if (episode_steps//2)-1 < len(self.processed_target):
                    if (test_answer(self.ins_decom, self.processed_target[(episode_steps//2)-1])):
                        reward = 0.3
            previous_string_idx, state, self.answer = test_model(self.args, self.model_decom, self.tokenizer, self.ins_decom)
        
        return state, reward, done, previous_string_idx, self.answer, flag


    def render(self):
        pass