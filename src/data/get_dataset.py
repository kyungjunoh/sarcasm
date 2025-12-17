from .dataset import KoCoSaDataset
from transformers import AutoTokenizer
import json
import pandas as pd

def load_kocosa(path):
    # jsonl 파일을 한 줄씩 직접 읽어서 파싱
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:      # 빈 줄 건너뛰기
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 문제가 있는 라인은 스킵 (필요하면 로그 찍어도 됨)

    df = pd.DataFrame(records)

    # 컬럼 이름 실제 데이터에 맞게 사용
    df['full_text'] = df['context'] + " [SEP] " + df['response']
    df['label'] = df['label'].apply(lambda x: 0 if "Non" in str(x) else 1)
    return df

def get_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_df = load_kocosa(args.train_df)
    val_df = load_kocosa(args.val_df)
    test_df = load_kocosa(args.test_df)

    train_dataset = KoCoSaDataset(train_df, tokenizer=tokenizer)
    val_dataset = KoCoSaDataset(val_df, tokenizer=tokenizer)
    test_dataset = KoCoSaDataset(test_df, tokenizer=tokenizer)

    return train_dataset, val_dataset, test_dataset