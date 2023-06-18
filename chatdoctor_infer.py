import datetime
import gzip
import logging
import os, json
import pickle
from gpt_infer import load_parser_and_args, init_logger, Accuracy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time

model = None
tokenizer = None
generator = None
os.environ["CUDA_VISIBLE_DEVICES"]="0"
logger = logging.getLogger('logger')


def load_model(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.LLaMATokenizer.from_pretrained(model_name)
    model = transformers.LLaMAForCausalLM.from_pretrained(
        model_name,
        #device_map=device_map,
        #device_map="auto",
        torch_dtype=torch.float16,
        #max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
        #load_in_8bit=eight_bit,
        #from_tf=True,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()

    generator = model.generate

load_model("/data/intern/sblee/chatdoctor_pretrained")

#First_chat = "ChatDoctor: I am ChatDoctor, what medical questions do you have?"

invitation = "ChatDoctor: "
human_invitation = "Patient: "


def Infer(msg):

    fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + human_invitation + msg + "\n\n" + invitation
    #fulltext = "\n\n".join(history) + "\n\n" + invitation
    
    logger.info('SENDING==========')
    logger.info(fulltext)
    logger.info('==========')

    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
    in_tokens = len(gen_in)
    with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=200,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5, # default: 1.0
                top_k = 50, # default: 50
                top_p = 1.0, # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

            text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt

    response = response.split(human_invitation)[0]

    response.strip()

    print(invitation + response) # ChatDoctor: ...

    print("")

    return response



def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
 
    # setup logging
    init_logger(args)
    
    # load data
    with gzip.open(args.task_dir, 'r') as f:
        fr = pickle.load(f)
    prediction = dict()

    # model
    model = Infer
  
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
        model_input = prompt.format(**input_dict)

        logger.info("***** Model Input *****")
        logger.info(model_input)

        # inference
        pred = model(model_input)
        logger.info("***** Model Output *****")
        logger.info({input_dict['label'] : pred})

        # saving
        prediction[idx] = {'groundtruth' : input_dict['label'] , 'prediction' : pred}
 
    # accuracy
    accr = Accuracy(prediction)
    logger.info("accuracy: {}".format(accr))

    result = {'accuracy' : accr, 'predictions' : prediction }

    with open(os.path.join(args.output_dir, 'chatdoctor_{}_{:%Y-%m-%d-%H:%M:%S}_predicted_results.json'.format(args.prompt, datetime.datetime.now())), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return accr

if __name__ == "__main__":
    parser, args = load_parser_and_args()
    main(args)