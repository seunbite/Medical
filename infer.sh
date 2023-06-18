#CUDA_VISIBLE_DEVICES=0,1,2 python llama_infer.py

CUDA_LAUNCH_BLOCKING=1 python llama_infer.py

CUDA_LAUNCH_BLOCKING=1 python gpt_infer.py

CUDA_LAUNCH_BLOCKING=1 python chatdoctor_infer.py
