import os
import sys
import warnings
from transformers import TrainingArguments
from transformers import Trainer
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import random
import numpy as np
import torch

from configs.train_arguements import get_arguments

from src.data.get_dataset import get_dataset
from src.model.get_model import get_model
from src.utils.compute_metrics import compute_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device : {device}")
    set_seed(42)

    ## =================== Data =================== ##
    train_dataset, val_dataset, test_dataset = get_dataset(args)

    ## =================== Base_Model =================== ##
    model = get_model(args)
    model.to(device)

    ## =================== Trainer =================== ##


    training_args = TrainingArguments(
        output_dir=f"./ckpt/{args.model}",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        load_best_model_at_end=True
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

    # gpt_prompt 모델은 few-shot 프롬프트 기반이므로 학습하지 않고 평가만 수행
    if args.model == 'gpt_prompt':
        print("\nFew-shot 프롬프트 모델 - 학습 없이 평가만 수행")
        print("\nValidation 결과:")
        val_metrics = trainer.evaluate(eval_dataset=val_dataset)
        print(val_metrics)
        
        print("\nTest 결과:")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        print(test_metrics)
    else:
        # 다른 모델들은 정상적으로 학습
        trainer.train()
        
        print("\n최종 테스트 결과 평가:")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        print(metrics)

if __name__=="__main__":
    args = get_arguments()   
    main(args)
