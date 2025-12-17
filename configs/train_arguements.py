from argparse import ArgumentParser
import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # 한 단계 상위 디렉토리
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_arguments():
    """
    Command line argument parser
    """
    parser = ArgumentParser(description="Training Arguments")
    
    # Dataset
    parser.add_argument("--train_df", type=str, default="src/data/KoCoSa_json/train.jsonl", help="Path to the training data directory")
    parser.add_argument("--val_df", type=str, default="src/data/KoCoSa_json/val.jsonl", help="Path to the validation data directory")
    parser.add_argument("--test_df", type=str, default="src/data/KoCoSa_json/test.jsonl", help="Path to the test data directory")
    

    # Training parameters
    parser.add_argument("--model", type=str, default="bert", help="Model type to use for training")
    parser.add_argument("--model_name", type=str, default="skt/kobert-base-v1", help="Pretrained model name or path")
    parser.add_argument("--eval_steps", type=int, default=200, help="Number of steps between evaluations")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs to train the model")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")


    parser.add_argument("--save_strategy", type=str, default='epoch', help="Save strategy: 'no', 'steps', 'epoch'")
    parser.add_argument("--eval_strategy", type=str, default='epoch', help="Eval strategy: 'no', 'steps', 'epoch'")

    return parser.parse_args()
