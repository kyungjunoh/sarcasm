#bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py \
    --model 'gpt' \
    --model_name 'EleutherAI/polyglot-ko-1.3b'
    # --model 'bert' \
    # --model_name 'skt/kobert-base-v1'