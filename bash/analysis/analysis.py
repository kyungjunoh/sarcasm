import os
import json
import pandas as pd
from tqdm import tqdm
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import refactored classes from the analyzers package
from src.analyzers import (
    MorphologyAnalyzer,
    SyntaxAnalyzer,
    PragmaticAnalyzer,
    IntegratedAnalyzer,
    utils
)

# Enable pandas integration for tqdm
tqdm.pandas()


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
    DATA_DIR = '/data/deep/sarcasm/src/data/KoCoSa_json'
    df = load_kocosa_data(DATA_DIR)
    df = preprocess_data(df)

    texts = df['response'].tolist()
    labels = df['label'].tolist()
    morphs_list = df['response_morphs'].tolist()
    pos_tags_list = df['response_pos'].tolist()

    # --- 2. Individual Analyzer Execution ---
    
    # Morphological Analysis
    print("\n" + "-"*70)
    print("1️⃣  Running Morphological Analysis")
    print("-"*70)
    morphology_analyzer = MorphologyAnalyzer()
    morphology_analyzer.analyze(pos_tags_list, labels)
    morphology_analyzer.visualize()

    # Syntactic Analysis
    print("\n" + "-"*70)
    print("2️⃣  Running Syntactic Analysis")
    print("-"*70)
    syntax_analyzer = SyntaxAnalyzer()
    syntax_analyzer.analyze(pos_tags_list, labels)
    syntax_analyzer.visualize()

    # Pragmatic Analysis
    print("\n" + "-"*70)
    print("3️⃣  Running Pragmatic Analysis")
    print("-"*70)
    pragmatic_analyzer = PragmaticAnalyzer()
    pragmatic_analyzer.analyze(texts, labels)
    pragmatic_analyzer.visualize()

    # Semantic Analysis (Reference)
    # Semantic analysis requires using a pre-trained language model (e.g., BERT)
    # to extract embeddings of context and response, and calculating sentiment scores
    # through a sentiment analysis model. Therefore, it is not run by default in this script.
    # If you have the relevant data (embeddings, sentiment scores), you can run it as follows.
    #
    # print("\n" + "-"*70)
    # print("4️⃣  Semantic Analysis (Not Run)")
    # print("-"*70)
    # print("💡 Semantic analysis requires embedding and sentiment score data.")
    # semantic_analyzer = SemanticAnalyzer()
    # semantic_analyzer.analyze(context_embeddings, response_embeddings, context_sentiments, response_sentiments, labels)
    # semantic_analyzer.visualize()
    
    # --- 3. Integrated Analysis ---
    print("\n" + "-"*70)
    print("📈 Running Integrated Analysis")
    print("-"*70)
    
    # Pass SemanticAnalyzer as None since data is not available
    integrated_analyzer = IntegratedAnalyzer(
        morphology=morphology_analyzer,
        syntax=syntax_analyzer,
        pragmatic=pragmatic_analyzer,
        semantic=None  
    )
    integrated_analyzer.visualize()

    print("\n✅ All analyses are complete!")
    print("📁 Generated analysis image files:")
    print("   - morphology_analysis.png")
    print("   - syntax_analysis.png")
    print("   - pragmatic_analysis.png")
    print("   - integrated_analysis_summary.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    # If `tqdm` is not in `requirements.txt`, you may need to install it.
    # pip install tqdm
    main()