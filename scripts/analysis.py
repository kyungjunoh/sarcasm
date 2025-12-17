import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.insert(0, PARENT_DIR)

# Import refactored classes from the analyzers package
from src.analyzers import (
    MorphologyAnalyzer,
    SyntaxAnalyzer,
    SemanticAnalyzer,
    utils
)

# Enable pandas integration for tqdm
tqdm.pandas()

# --- Helper functions for Semantic Analysis ---

def get_embeddings(texts: List[str], model, tokenizer, device, batch_size=32) -> np.ndarray:
    """Computes embeddings for a list of texts using a sentence-transformer model."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform pooling (mean pooling)
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        
        all_embeddings.append(sentence_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def get_sentiments(texts: List[str], sentiment_pipeline, batch_size=32) -> List[float]:
    """Computes sentiment scores for a list of texts."""
    sentiments = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
        batch = texts[i:i+batch_size]
        results = sentiment_pipeline(batch)
        for res in results:
            score = res['score'] if res['label'] == 'positive' else -res['score']
            sentiments.append(score)
    return sentiments

# --- Main Analysis Pipeline ---

def load_kocosa_data(data_dir: str) -> pd.DataFrame:
    """
    Reads all .jsonl files from the KoCoSa dataset directory and loads them into a DataFrame.
    Combines train, validation, and test files.
    """
    all_data = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    
    print(f"📂 Loading data: {', '.join(files)}")
    
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))
    
    df = pd.DataFrame(all_data)
    
    # Convert labels to integers (Sarcasm: 1, Non-Sarcasm: 0)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'Sarcasm' else 0)
    
    print(f"✅ Total {len(df)} samples loaded.")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by applying morphological analysis and part-of-speech tagging.
    """
    if not utils.okt:
        print("❗️ konlpy is not installed. Skipping morphological/syntactic analysis.")
        df['response_morphs'] = df['response'].apply(lambda x: x.split())
        df['response_pos'] = df['response'].apply(lambda x: [])
        return df

    print("\n🔄 Preprocessing data (morphological analysis)...")
    
    # Use progress_apply to show progress
    df['response_morphs'] = df['response'].progress_apply(utils.get_morphs)
    df['response_pos'] = df['response'].progress_apply(utils.get_pos)
    
    print("✅ Preprocessing complete.")
    return df


def main():
    """
    Runs the entire analysis pipeline.
    """
    print("="*70)
    print("📊 Linguistic Feature Analysis of Korean Sarcasm")
    print("="*70)

    # --- 1. Data Loading and Preprocessing ---
    DATA_DIR = '/Users/kipyokim/Desktop/langcont/data/KoCoSa_json'
    df = load_kocosa_data(DATA_DIR)
    # Reduce dataset size for faster processing if needed
    # df = df.sample(n=1000, random_state=42).copy()
    
    df = preprocess_data(df)

    texts = df['response'].tolist()
    labels = df['label'].tolist()
    morphs_list = df['response_morphs'].tolist()
    pos_tags_list = df['response_pos'].tolist()

    # --- 2. Analyzers Execution ---
    
    # Morphological Analysis
    print("\n" + "-"*70)
    print("1️⃣  Running Morphological Analysis")
    print("-"*70)
    morphology_analyzer = MorphologyAnalyzer()
    morphology_analyzer.analyze(morphs_list, labels)
    morphology_analyzer.visualize()

    # Syntactic Analysis
    print("\n" + "-"*70)
    print("2️⃣  Running Syntactic Analysis")
    print("-"*70)
    syntax_analyzer = SyntaxAnalyzer()
    syntax_analyzer.analyze(pos_tags_list, labels)
    syntax_analyzer.visualize()

    # Semantic Analysis
    print("\n" + "-"*70)
    print("4️⃣  Running Semantic Analysis")
    print("-"*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    # Load models
    print("🧠 Loading models for semantic analysis...")
    embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1')
    embedding_model = AutoModel.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1').to(device)
    sentiment_pipeline = pipeline('sentiment-analysis', model='matthewburke/korean_sentiment', device=0 if device.type == 'cuda' else -1)
    print("✅ Models loaded.")

    # Get embeddings and sentiments
    context_texts = df['context'].tolist()
    response_texts = df['response'].tolist()

    context_embeddings = get_embeddings(context_texts, embedding_model, embedding_tokenizer, device)
    response_embeddings = get_embeddings(response_texts, embedding_model, embedding_tokenizer, device)
    context_sentiments = get_sentiments(context_texts, sentiment_pipeline)
    response_sentiments = get_sentiments(response_texts, sentiment_pipeline)
    
    semantic_analyzer = SemanticAnalyzer()
    semantic_analyzer.analyze(context_embeddings, response_embeddings, context_sentiments, response_sentiments, labels)
    semantic_analyzer.visualize()

    print("\n✅ All analyses are complete!")
    print("📁 Generated analysis image files:")
    print("   - morphology_analysis.png")
    print("   - syntax_analysis.png")
    print("   - semantic_analysis.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()