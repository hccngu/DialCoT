import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
model_path='model/llama_7b'
check_path='model/llama_7b_DIAL'        #这个是dial-CoT-final-3epoch的保存目录
max_new_tokens = 512
generation_config = dict(
    temperature=1.0,
    top_k=30,
    top_p=1,
    do_sample=False,
    num_return_sequences=3,
    num_beams=3,
    repetition_penalty=1.2,
    output_hidden_states=True,
    num_beam_groups=3,
    diversity_penalty=1.5,          #如果生成重复较多，可以调整这个参数
    max_new_tokens=max_new_tokens
)
load_type = torch.float32 #Sometimes may need torch.float32
device = torch.device('cpu')
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"
model_config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(check_path, torch_dtype=load_type, config=model_config)
model.to(device)
model.eval()
print("Load model successfully")
text="Based on the given text, generate the next question that needs to be answered.\nContent: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?\nDecomposer: "
inputs = tokenizer('Human: \n'+text+'\n\nAssistant:\n', max_length=max_new_tokens,truncation=True,return_tensors="pt")
print(inputs)
generation_output = model.generate(
            input_ids = inputs["input_ids"].to(device), 
            **generation_config
    )

print(generation_output)



result= tokenizer.batch_decode(generation_output,skip_special_tokens=True)              #由于没有输出attention_mask，先decode再encode
state_inputs=tokenizer(result, max_length=max_new_tokens,padding=True,truncation=True,return_tensors="pt").to(device)
# f=open('abc.txt', 'w')
state=model(**state_inputs,output_hidden_states=True).hidden_states    #33,(3,seq_len,4096)

last_hidden_states=state[-1]
print(last_hidden_states.shape)


