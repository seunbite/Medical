import datetime
import gzip
import logging
import os, json
import gc
import pickle
from gpt_infer import load_parser_and_args, init_logger, Accuracy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
import transformers
import torch
import torch.nn as nn
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map
import accelerate
import time

logger = logging.getLogger('logger')
# MODEL = 'decapoda-research/llama-7b-hf'
MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'
model_name = MODEL.split('/')[-1]


def load_model(MODEL):
    if os.path.exists('./trained'):
        MODEL = './trained'

    tokenizer = transformers.LLaMATokenizer.from_pretrained(MODEL)
    #model = transformers.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, device_map="auto")
    model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            #load_in_8bit=eight_bit,
            #from_tf=True,
            low_cpu_mem_usage=True,
            #load_in_8bit=False,
            cache_dir="cache"
        )

    # will use 6 Gb of GPU VRAM, others to CPU RAM
    device_map = infer_auto_device_map(model, max_memory={0: "20GiB", 1: "20GiB", 2: "20GiB"})
    print(device_map)

    return tokenizer, model

# batch = tokenizer("The highest mountain in China is ", return_tensors="pt")
# print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), do_sample=True, top_k=50, max_length=100, top_p=0.95, temperature=1.0)[0]))



def main(args, MODEL):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
 
    # setup logging
    init_logger(args)
    
    # load data
    with gzip.open(args.task_dir, 'r') as f:
        fr = pickle.load(f)
    prediction = dict()

    # model
    tokenizer, model = load_model(MODEL)
  
    # input
    for t in fr:
        idx = t['file name']
        input_dict = {}
        name_list = ['label', 'age', 'sex', 'educ', 'marriage', 'apoe', 'mmse', 'fmri']
        for name in name_list:
            input_dict[name] = t[name]
        for id in range(22):
            input_dict['q{}'.format(id)] = t['q{}'.format(id)]

        # prompt making
        with open(os.path.join(args.base_dir, 'prompt', '{}.json'.format(args.prompt)), 'r') as f:
            prompt = f.read()
        input = prompt.format(**input_dict)

        logger.info("***** Model Input *****")
        logger.info(input)

        # inference
        model_input = tokenizer(input, return_tensors="pt")
        pred = tokenizer.decode(model.generate(model_input["input_ids"].cuda(), 
                                               do_sample=True, 
                                               top_k=50, 
                                               max_length=100, 
                                               top_p=0.95, 
                                               temperature=1.0)[0])

        logger.info("***** Model Output *****")
        logger.info({input_dict['label'] : pred})

        # saving
        prediction[idx] = {'groundtruth' : input_dict['label'] , 'prediction' : pred}
 
    # accuracy
    accr = Accuracy(prediction)
    logger.info("accuracy: {}".format(accr))

    result = {'accuracy' : accr, 'predictions' : prediction }

    with open(os.path.join(args.output_dir, '{}_{}_{:%Y-%m-%d-%H:%M:%S}_predicted_results.json'.format(model_name, args.prompt, datetime.datetime.now())), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return accr 

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    parser, args = load_parser_and_args()
    main(args, MODEL)