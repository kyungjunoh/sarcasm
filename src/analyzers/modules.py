import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple
import warnings
import re
import platform

from .base import BaseAnalyzer
import koreanize_matplotlib


if platform.system() == 'Darwin':  
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows': 
    plt.rcParams['font.family'] = 'Malgun Gothic'


plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class MorphologyAnalyzer(BaseAnalyzer):
    """Morphological analysis class - analyzes which morphemes appear frequently"""
    
    def __init__(self):
        self.sarcasm_morphemes = Counter()
        self.normal_morphemes = Counter()
        self.sarcasm_count = 0
        self.normal_count = 0
        
    def analyze(self, morphs_list: List[List[str]], labels: List[int]):
        """
        Analyze morpheme frequency
        
        Args:
            morphs_list: List of morpheme analysis results
            labels: Labels (0: normal, 1: sarcasm)
        """
        for morphs, label in zip(morphs_list, labels):
            if label == 1:  
                self.sarcasm_morphemes.update(morphs)
                self.sarcasm_count += 1
            else:  #
                self.normal_morphemes.update(morphs)
                self.normal_count += 1
        
        print(f" Morphological analysis complete: {self.sarcasm_count} sarcasm, {self.normal_count} normal")
        
    def get_distinctive_features(self, top_n: int = 20) -> pd.DataFrame:
        """Extract distinctive sarcastic morphemes"""
        

        sarcasm_freq = {k: v/self.sarcasm_count for k, v in self.sarcasm_morphemes.items()}
        normal_freq = {k: v/self.normal_count for k, v in self.normal_morphemes.items()}
        

        all_morphemes = set(sarcasm_freq.keys()) | set(normal_freq.keys())
        differences = []
        
        for morph in all_morphemes:
            s_freq = sarcasm_freq.get(morph, 0)
            n_freq = normal_freq.get(morph, 0)
            diff = s_freq - n_freq
            
            if s_freq > 0.01:  
                differences.append({
                    'morpheme': morph,
                    'sarcasm_freq': s_freq * 100,
                    'normal_freq': n_freq * 100,
                    'difference': diff * 100,
                    'sarcasm_ratio': s_freq / (s_freq + n_freq) * 100 if (s_freq + n_freq) > 0 else 0
                })
        
        df = pd.DataFrame(differences)
        df = df.sort_values('difference', ascending=False).head(top_n)
        
        return df
    
    def visualize(self, top_n: int = 15):
        """Visualize morphology analysis results"""
        
        df = self.get_distinctive_features(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
    
        x = np.arange(len(df))
        width = 0.35
        
        axes[0].bar(x - width/2, df['sarcasm_freq'], width, label='Sarcasm', color='#ef4444', alpha=0.8)
        axes[0].bar(x + width/2, df['normal_freq'], width, label='Normal', color='#10b981', alpha=0.8)
        axes[0].set_xlabel('Morpheme', fontsize=12)
        axes[0].set_ylabel('Frequency (%)', fontsize=12)
        axes[0].set_title('Morpheme Frequency Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df['morpheme'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
      
        colors = plt.cm.RdYlGn_r(df['sarcasm_ratio'] / 100)
        axes[1].barh(df['morpheme'], df['sarcasm_ratio'], color=colors, alpha=0.8)
        axes[1].set_xlabel('Sarcasm Sentence Ratio (%)', fontsize=12)
        axes[1].set_title('Sarcastic Signal Morphemes (Higher is Stronger)', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
     
        axes[1].axvline(x=80, color='red', linestyle='--', alpha=0.5, label='80% Threshold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('morphology_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        
        print("\n Key Findings in Morphological Analysis:")
        print(f"  • Strongest sarcastic signal: '{df.iloc[0]['morpheme']}' (Sarcasm occurrence rate {df.iloc[0]['sarcasm_ratio']:.1f}%)")
        print(f"  • Top 3 sarcasm frequency morphemes: {', '.join(df.head(3)['morpheme'].tolist())}")
        print(f"  • Morphemes with >= 80% sarcastic signal: {len(df[df['sarcasm_ratio'] >= 80])}")


class SyntaxAnalyzer(BaseAnalyzer):
    """Syntactic analysis class - analyzes sentence structure patterns"""
    
    def __init__(self):
        self.sarcasm_patterns = []
        self.normal_patterns = []
        self.sarcasm_lengths = []
        self.normal_lengths = []
        
    def analyze(self, pos_tags_list: List[List[str]], labels: List[int]):
        """
        Analyze syntactic patterns
        
        Args:
            pos_tags_list: List of part-of-speech tag sequences
            labels: Labels (0: normal, 1: sarcasm)
        """
        for pos_tags, label in zip(pos_tags_list, labels):
            
            trigrams = [f"{pos_tags[i]}-{pos_tags[i+1]}-{pos_tags[i+2]}" for i in range(len(pos_tags)-2)]
            
            if label == 1:  
                self.sarcasm_patterns.extend(trigrams)
                self.sarcasm_lengths.append(len(pos_tags))
            else:  
                self.normal_patterns.extend(trigrams)
                self.normal_lengths.append(len(pos_tags))
        
        print(f" Syntactic analysis complete: {len(self.sarcasm_lengths)} sarcasm, {len(self.normal_lengths)} normal")
    
    def get_distinctive_features(self, top_n: int = 15) -> pd.DataFrame:
        """Extract distinctive syntactic patterns"""
        
        sarcasm_counter = Counter(self.sarcasm_patterns)
        normal_counter = Counter(self.normal_patterns)
        
        sarcasm_total = len(self.sarcasm_patterns)
        normal_total = len(self.normal_patterns)
        
        all_patterns = set(sarcasm_counter.keys()) | set(normal_counter.keys())
        differences = []
        
        for pattern in all_patterns:
            s_freq = sarcasm_counter[pattern] / sarcasm_total if sarcasm_total > 0 else 0
            n_freq = normal_counter[pattern] / normal_total if normal_total > 0 else 0
            diff = s_freq - n_freq
            
            if s_freq > 0.005:  
                differences.append({
                    'pattern': pattern,
                    'sarcasm_freq': s_freq * 100,
                    'normal_freq': n_freq * 100,
                    'difference': diff * 100
                })
        
        df = pd.DataFrame(differences)
        df = df.sort_values('difference', ascending=False).head(top_n)
        
        return df
    
    def visualize(self):
        """Visualize syntax analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        
        df_patterns = self.get_distinctive_features(12)
        x = np.arange(len(df_patterns))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, df_patterns['sarcasm_freq'], width, label='Sarcasm', color='#8b5cf6', alpha=0.8)
        axes[0, 0].bar(x + width/2, df_patterns['normal_freq'], width, label='Normal', color='#06b6d4', alpha=0.8)
        axes[0, 0].set_xlabel('POS Pattern', fontsize=12)
        axes[0, 0].set_ylabel('Frequency (%)', fontsize=12)
        axes[0, 0].set_title('Distinctive Syntactic Patterns (Trigram)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(df_patterns['pattern'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        
        axes[0, 1].hist(self.sarcasm_lengths, bins=20, alpha=0.6, label='Sarcasm', color='#ef4444', edgecolor='black')
        axes[0, 1].hist(self.normal_lengths, bins=20, alpha=0.6, label='Normal', color='#10b981', edgecolor='black')
        axes[0, 1].axvline(np.mean(self.sarcasm_lengths), color='#ef4444', linestyle='--', linewidth=2, label=f'Sarcasm Avg: {np.mean(self.sarcasm_lengths):.1f}')
        axes[0, 1].axvline(np.mean(self.normal_lengths), color='#10b981', linestyle='--', linewidth=2, label=f'Normal Avg: {np.mean(self.normal_lengths):.1f}')
        axes[0, 1].set_xlabel('Sentence Length (tokens)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Sentence Length Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        
        data_length = [self.sarcasm_lengths, self.normal_lengths]
        bp = axes[1, 0].boxplot(data_length, labels=['Sarcasm', 'Normal'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#ef4444')
        bp['boxes'][1].set_facecolor('#10b981')
        axes[1, 0].set_ylabel('Sentence Length (tokens)', fontsize=12)
        axes[1, 0].set_title('Sentence Length Comparison (Boxplot)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
    
        df_top = df_patterns.head(10)
        axes[1, 1].barh(df_top['pattern'], df_top['difference'], color='#f59e0b', alpha=0.8)
        axes[1, 1].set_xlabel('Frequency Difference (Sarcasm - Normal) %', fontsize=12)
        axes[1, 1].set_title('Sarcastic Syntactic Patterns (Difference)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('syntax_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    
        print("\n Key Findings in Syntactic Analysis:")
        print(f"  • Avg sentence length (Sarcasm): {np.mean(self.sarcasm_lengths):.1f} tokens")
        print(f"  • Avg sentence length (Normal): {np.mean(self.normal_lengths):.1f} tokens")
        print(f"  • Length difference: {np.mean(self.normal_lengths) - np.mean(self.sarcasm_lengths):.1f} tokens (Normal is longer)")
        print(f"  • Most distinctive pattern: '{df_patterns.iloc[0]['pattern']}'")


class SemanticAnalyzer(BaseAnalyzer):
    """Semantic analysis class - analyzes semantic mismatch between context and response"""
    
    def __init__(self):
        self.similarities = []
        self.context_sentiments = []
        self.response_sentiments = []
        self.labels = []
        
    def analyze(self, context_embeddings: np.ndarray, response_embeddings: np.ndarray, 
                context_sentiments: List[float], response_sentiments: List[float], 
                labels: List[int]):
        """
        Analyze semantic similarity and sentiment polarity
        
        Args:
            context_embeddings: Context embeddings
            response_embeddings: Response embeddings
            context_sentiments: Context sentiment scores (-1 to 1)
            response_sentiments: Response sentiment scores (-1 to 1)
            labels: Labels
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        for i in range(len(labels)):
            sim = cosine_similarity(
                context_embeddings[i].reshape(1, -1),
                response_embeddings[i].reshape(1, -1)
            )[0][0]
            
            self.similarities.append(sim)
            self.context_sentiments.append(context_sentiments[i])
            self.response_sentiments.append(response_sentiments[i])
            self.labels.append(labels[i])
        
        print(f"✅ Semantic analysis complete: {len(self.labels)} samples total")
    
    def get_statistics(self) -> Dict:
        """Statistical summary"""
        sarcasm_sims = [s for s, l in zip(self.similarities, self.labels) if l == 1]
        normal_sims = [s for s, l in zip(self.similarities, self.labels) if l == 0]
        
        sarcasm_polarity_gaps = [
            abs(c - r) for c, r, l in zip(self.context_sentiments, self.response_sentiments, self.labels) if l == 1
        ]
        normal_polarity_gaps = [
            abs(c - r) for c, r, l in zip(self.context_sentiments, self.response_sentiments, self.labels) if l == 0
        ]
        
        return {
            'sarcasm_similarity_mean': np.mean(sarcasm_sims) if sarcasm_sims else 0,
            'normal_similarity_mean': np.mean(normal_sims) if normal_sims else 0,
            'sarcasm_polarity_gap_mean': np.mean(sarcasm_polarity_gaps) if sarcasm_polarity_gaps else 0,
            'normal_polarity_gap_mean': np.mean(normal_polarity_gaps) if normal_polarity_gaps else 0,
        }
    
    def visualize(self):

        
        stats = self.get_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        
        sarcasm_sims = [s for s, l in zip(self.similarities, self.labels) if l == 1]
        normal_sims = [s for s, l in zip(self.similarities, self.labels) if l == 0]
        
        axes[0, 0].hist(sarcasm_sims, bins=30, alpha=0.6, label='Sarcasm', color='#ef4444', edgecolor='black')
        axes[0, 0].hist(normal_sims, bins=30, alpha=0.6, label='Normal', color='#10b981', edgecolor='black')
        axes[0, 0].axvline(np.mean(sarcasm_sims), color='#ef4444', linestyle='--', linewidth=2)
        axes[0, 0].axvline(np.mean(normal_sims), color='#10b981', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Cosine Similarity', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Context-Response Semantic Similarity Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
      
        sarcasm_mask = np.array(self.labels) == 1
        axes[0, 1].scatter(
            np.array(self.context_sentiments)[sarcasm_mask],
            np.array(self.response_sentiments)[sarcasm_mask],
            c='#ef4444', alpha=0.5, label='Sarcasm', s=30
        )
        axes[0, 1].scatter(
            np.array(self.context_sentiments)[~sarcasm_mask],
            np.array(self.response_sentiments)[~sarcasm_mask],
            c='#10b981', alpha=0.5, label='Normal', s=30
        )
        axes[0, 1].plot([-1, 1], [-1, 1], 'k--', alpha=0.3, label='Identity Line')
        axes[0, 1].set_xlabel('Context Sentiment Score', fontsize=12)
        axes[0, 1].set_ylabel('Response Sentiment Score', fontsize=12)
        axes[0, 1].set_title('Sentiment Polarity Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_xlim(-1, 1)
        axes[0, 1].set_ylim(-1, 1)
  
        polarity_gaps_s = [abs(c - r) for c, r, l in zip(self.context_sentiments, self.response_sentiments, self.labels) if l == 1]
        polarity_gaps_n = [abs(c - r) for c, r, l in zip(self.context_sentiments, self.response_sentiments, self.labels) if l == 0]
        
        data = [polarity_gaps_s, polarity_gaps_n]
        bp = axes[1, 0].boxplot(data, labels=['Sarcasm', 'Normal'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#ef4444')
        bp['boxes'][1].set_facecolor('#10b981')
        axes[1, 0].set_ylabel('Sentiment Polarity Gap (abs)', fontsize=12)
        axes[1, 0].set_title('Degree of Sentiment Polarity Mismatch', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        categories = ['Semantic Similarity', 'Polarity Gap']
        sarcasm_values = [stats['sarcasm_similarity_mean'], stats['sarcasm_polarity_gap_mean']]
        normal_values = [stats['normal_similarity_mean'], stats['normal_polarity_gap_mean']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, sarcasm_values, width, label='Sarcasm', color='#ef4444', alpha=0.8)
        axes[1, 1].bar(x + width/2, normal_values, width, label='Normal', color='#10b981', alpha=0.8)
        axes[1, 1].set_ylabel('Average Value', fontsize=12)
        axes[1, 1].set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('semantic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
      
        print("\n📊 Key Findings in Semantic Analysis:")
        print(f"  • Avg similarity (Sarcasm): {stats['sarcasm_similarity_mean']:.3f}")
        print(f"  • Avg similarity (Normal): {stats['normal_similarity_mean']:.3f}")
        print(f"  • Similarity difference: {stats['normal_similarity_mean'] - stats['sarcasm_similarity_mean']:.3f} (Normal is higher)")
        print(f"  • Avg polarity gap (Sarcasm): {stats['sarcasm_polarity_gap_mean']:.3f}")
        print(f"  • Avg polarity gap (Normal): {stats['normal_polarity_gap_mean']:.3f}")
        print(f"  ⚠️  Sarcasm shows lower semantic similarity and a larger polarity gap!")


class PragmaticAnalyzer(BaseAnalyzer):
    """Pragmatic analysis class - analyzes literal meaning vs. intended meaning in context"""
    
    def __init__(self):
        self.sarcasm_cases = []  
        self.normal_cases = []   
        
       
        self.intention_patterns = Counter() 
        self.context_response_alignment = []
        
    def analyze(self, contexts: List[str], responses: List[str], explanations: List[str], 
                context_morphs: List[List[str]], response_morphs: List[List[str]], 
                explanation_morphs: List[List[str]], labels: List[int],
                context_sentiments: List[float] = None, response_sentiments: List[float] = None):
    
        for idx, (ctx, resp, exp, ctx_m, resp_m, exp_m, label) in enumerate(zip(
            contexts, responses, explanations, context_morphs, response_morphs, explanation_morphs, labels
        )):
            ctx_sent = context_sentiments[idx] if context_sentiments else 0
            resp_sent = response_sentiments[idx] if response_sentiments else 0
            
            if label == 1:  
                self.sarcasm_cases.append({
                    'context': ctx,
                    'response': resp,
                    'explanation': exp,
                    'context_morphs': ctx_m,
                    'response_morphs': resp_m,
                    'explanation_morphs': exp_m,
                    'context_sentiment': ctx_sent,
                    'response_sentiment': resp_sent,
                    'sentiment_mismatch': self._detect_sentiment_mismatch(ctx_sent, resp_sent)
                })
                
                
                if exp_m:
                    intentions = self._extract_intentions(exp, exp_m)
                    self.intention_patterns.update(intentions)
                    
            else: 
                self.normal_cases.append({
                    'context': ctx,
                    'response': resp,
                    'context_morphs': ctx_m,
                    'response_morphs': resp_m,
                    'context_sentiment': ctx_sent,
                    'response_sentiment': resp_sent,
                    'sentiment_mismatch': self._detect_sentiment_mismatch(ctx_sent, resp_sent)
                })
        
        print(f"✅ Pragmatic analysis complete: {len(self.sarcasm_cases)} sarcasm (meaning mismatch), {len(self.normal_cases)} normal cases")
    
    def _detect_sentiment_mismatch(self, context_sent: float, response_sent: float) -> str:
        """Detect if response sentiment mismatches with context sentiment"""
        # Positive context + Negative response or vice versa = mismatch
        if context_sent > 0.1 and response_sent < -0.1:
            return 'positive_to_negative'
        elif context_sent < -0.1 and response_sent > 0.1:
            return 'negative_to_positive'
        elif abs(context_sent - response_sent) > 0.5:
            return 'large_gap'
        else:
            return 'aligned'
    
    def _extract_intentions(self, explanation: str, exp_morphs: List[str]) -> List[str]:
        """Extract what the speaker actually intends from sarcasm explanation"""
        intentions = []
        
        # Common intention keywords in Korean sarcasm explanations
        intention_keywords = {
            '비꼬': 'mocking',
            '조롱': 'ridicule',
            '반대': 'opposite_meaning',
            '위로': 'comfort',
            '공감': 'empathy',
            '비난': 'criticism',
            '칭찬': 'praise_opposite',
            '농담': 'joke',
            '장난': 'teasing',
            '아이러니': 'irony',
            '역설': 'paradox',
            '반어': 'irony',
            '빈정': 'sarcasm',
            '냉소': 'cynicism'
        }
        
        for keyword, intention in intention_keywords.items():
            if keyword in explanation:
                intentions.append(intention)
        
        # If no specific keyword found, mark as general sarcasm
        if not intentions:
            intentions.append('general_sarcasm')
        
        return intentions
    
    def get_distinctive_features(self) -> pd.DataFrame:
        """Analyze meaning mismatch patterns"""
        
        # Sentiment mismatch distribution
        sarcasm_mismatch = Counter([case['sentiment_mismatch'] for case in self.sarcasm_cases])
        normal_mismatch = Counter([case['sentiment_mismatch'] for case in self.normal_cases])
        
        total_sarcasm = len(self.sarcasm_cases)
        total_normal = len(self.normal_cases)
        
        mismatch_types = ['positive_to_negative', 'negative_to_positive', 'large_gap', 'aligned']
        results = []
        
        for mtype in mismatch_types:
            sarcasm_count = sarcasm_mismatch[mtype]
            normal_count = normal_mismatch[mtype]
            
            sarcasm_rate = (sarcasm_count / total_sarcasm * 100) if total_sarcasm > 0 else 0
            normal_rate = (normal_count / total_normal * 100) if total_normal > 0 else 0
            
            results.append({
                'mismatch_type': mtype,
                'sarcasm_rate': sarcasm_rate,
                'normal_rate': normal_rate,
                'difference': sarcasm_rate - normal_rate
            })
        
        return pd.DataFrame(results)
    
    def get_intention_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """Analyze what sarcasm actually intends (from explanations)"""
        
        total = sum(self.intention_patterns.values())
        if total == 0:
            return pd.DataFrame()
        
        intentions = []
        for intent, count in self.intention_patterns.most_common(top_n):
            intentions.append({
                'intention': intent,
                'count': count,
                'percentage': count / total * 100
            })
        
        return pd.DataFrame(intentions)
    
    def get_statistics(self) -> Dict:
        """Calculate statistics on pragmatic meaning mismatch"""
        
        # Sentiment alignment analysis
        sarcasm_aligned = sum(1 for case in self.sarcasm_cases if case['sentiment_mismatch'] == 'aligned')
        normal_aligned = sum(1 for case in self.normal_cases if case['sentiment_mismatch'] == 'aligned')
        
        total_sarcasm = len(self.sarcasm_cases)
        total_normal = len(self.normal_cases)
        
        # Response sentiment distribution
        sarcasm_positive_resp = sum(1 for case in self.sarcasm_cases if case['response_sentiment'] > 0.1)
        sarcasm_negative_resp = sum(1 for case in self.sarcasm_cases if case['response_sentiment'] < -0.1)
        
        # Cases with explanations
        with_explanation = sum(1 for case in self.sarcasm_cases if case['explanation'])
        
        # Length statistics
        sarcasm_response_lengths = [len(case['response_morphs']) for case in self.sarcasm_cases]
        normal_response_lengths = [len(case['response_morphs']) for case in self.normal_cases]
        explanation_lengths = [len(case['explanation_morphs']) for case in self.sarcasm_cases if case['explanation_morphs']]
        
        return {
            'total_sarcasm': total_sarcasm,
            'total_normal': total_normal,
            'sarcasm_aligned_rate': (sarcasm_aligned / total_sarcasm * 100) if total_sarcasm > 0 else 0,
            'normal_aligned_rate': (normal_aligned / total_normal * 100) if total_normal > 0 else 0,
            'sarcasm_positive_response_rate': (sarcasm_positive_resp / total_sarcasm * 100) if total_sarcasm > 0 else 0,
            'sarcasm_negative_response_rate': (sarcasm_negative_resp / total_sarcasm * 100) if total_sarcasm > 0 else 0,
            'with_explanation': with_explanation,
            'explanation_rate': (with_explanation / total_sarcasm * 100) if total_sarcasm > 0 else 0,
            'avg_sarcasm_response_length': np.mean(sarcasm_response_lengths) if sarcasm_response_lengths else 0,
            'avg_normal_response_length': np.mean(normal_response_lengths) if normal_response_lengths else 0,
            'avg_explanation_length': np.mean(explanation_lengths) if explanation_lengths else 0
        }
    
    def visualize(self):
        
        
        df_intentions = self.get_intention_analysis()
        stats = self.get_statistics()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
    
        if not df_intentions.empty:
            intention_labels = {
                'mocking': '비꼬기 (Mocking)',
                'ridicule': '조롱 (Ridicule)',
                'opposite_meaning': '반대 의미 (Opposite)',
                'comfort': '위로 (Comfort)',
                'empathy': '공감 (Empathy)',
                'criticism': '비난 (Criticism)',
                'praise_opposite': '가짜 칭찬 (Fake Praise)',
                'joke': '농담 (Joke)',
                'teasing': '장난 (Teasing)',
                'irony': '아이러니 (Irony)',
                'general_sarcasm': '일반 비꼬기 (General)'
            }
            df_intentions['label'] = df_intentions['intention'].map(lambda x: intention_labels.get(x, x))
            
            colors_intent = plt.cm.Reds(df_intentions['percentage'] / df_intentions['percentage'].max())
            ax.barh(df_intentions['label'], df_intentions['percentage'], color=colors_intent, alpha=0.8)
            ax.set_xlabel('비율 (Percentage in Sarcasm Cases) %', fontsize=14)
            ax.set_title('비꼰기의 실제 의도 (Intended Meanings Behind Sarcasm)', fontsize=16, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
      
            for idx, (label, pct) in enumerate(zip(df_intentions['label'], df_intentions['percentage'])):
                ax.text(pct + 1, idx, f'{pct:.1f}%', va='center', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'No explanation data available', ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('pragmatic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
   
        print("\n 화용론적 분석 주요 결과 (Key Findings in Pragmatic Analysis):")
        print(f"  • 전체 비꼬기 사례: {stats['total_sarcasm']}개")
        print(f"  • 설명이 있는 사례: {stats['with_explanation']}개 ({stats['explanation_rate']:.1f}%)")
        
        if not df_intentions.empty:
            top_intention = df_intentions.iloc[0]
            print(f"\n  🎯 가장 흔한 비꼬기 의도: '{top_intention['label']}' ({top_intention['percentage']:.1f}%)")
            if len(df_intentions) > 1:
                print(f"  📊 상위 의도 분포:")
                for i, row in df_intentions.head(5).iterrows():
                    print(f"     - {row['label']}: {row['percentage']:.1f}%")
        
        print(f"\n  ⚠️  화용론적 핵심: 비꼬기는 '말한 것'과 '의도한 것'이 다릅니다!")
        print(f"      (Sarcasm shows clear pragmatic mismatch: literal ≠ intended meaning)")
    
    



class IntegratedAnalyzer:
    
    
    def __init__(self, morphology: MorphologyAnalyzer, syntax: SyntaxAnalyzer, pragmatic: PragmaticAnalyzer, semantic: SemanticAnalyzer = None):
        self.morphology = morphology
        self.syntax = syntax
        self.pragmatic = pragmatic
        self.semantic = semantic
    
    def visualize(self):

        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Morphology Summary
        ax1 = fig.add_subplot(gs[0, 0])
        morph_df = self.morphology.get_distinctive_features(8)
        ax1.barh(morph_df['morpheme'], morph_df['sarcasm_ratio'], color='#ef4444', alpha=0.8)
        ax1.set_xlabel('Sarcasm Ratio (%)', fontsize=10)
        ax1.set_title('Morphology: Sarcastic Signal Morphemes', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Syntax Summary
        ax2 = fig.add_subplot(gs[0, 1])
        sarcasm_len_mean = np.mean(self.syntax.sarcasm_lengths)
        normal_len_mean = np.mean(self.syntax.normal_lengths)
        ax2.bar(['Sarcasm', 'Normal'], [sarcasm_len_mean, normal_len_mean], 
                color=['#8b5cf6', '#06b6d4'], alpha=0.8)
        ax2.set_ylabel('Avg. Sentence Length (tokens)', fontsize=10)
        ax2.set_title('Syntax: Sentence Length Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for i, v in enumerate([sarcasm_len_mean, normal_len_mean]):
            ax2.text(i, v + 0.3, f'{v:.1f}', ha='center', fontweight='bold')
        
        # 3. Semantics Summary
        ax3 = fig.add_subplot(gs[0, 2])
        if self.semantic:
            stats = self.semantic.get_statistics()
            categories = ['Semantic\nSimilarity', 'Polarity\nGap']
            sarcasm_vals = [stats['sarcasm_similarity_mean'], stats['sarcasm_polarity_gap_mean']]
            normal_vals = [stats['normal_similarity_mean'], stats['normal_polarity_gap_mean']]
            
            x = np.arange(len(categories))
            width = 0.35
            ax3.bar(x - width/2, sarcasm_vals, width, label='Sarcasm', color='#ef4444', alpha=0.8)
            ax3.bar(x + width/2, normal_vals, width, label='Normal', color='#10b981', alpha=0.8)
            ax3.set_ylabel('Average Value', fontsize=10)
            ax3.set_title('Semantics: Key Metrics', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Semantic Analysis Skipped', ha='center', va='center', fontsize=12)
            ax3.set_title('Semantics: Key Metrics', fontsize=12, fontweight='bold')
            ax3.axis('off')

    
        ax4 = fig.add_subplot(gs[1, 0])
        prag_df = self.pragmatic.get_distinctive_features()
        signal_names_en = {
            'elongation': 'Elongation',
            'laughter': 'Laughter',
            'quotation': 'Quotation',
            'emphasis': 'Emphasis',
            'dot': 'Dots'
        }
        prag_df['signal_en'] = prag_df['signal'].map(signal_names_en)
        ax4.barh(prag_df['signal_en'], prag_df['difference'], 
                color=['#ef4444' if d > 0 else '#10b981' for d in prag_df['difference']], alpha=0.8)
        ax4.set_xlabel('Occurrence Rate Difference (%)', fontsize=10)
        ax4.set_title('Pragmatics: Sarcastic Signal Strength', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax4.grid(axis='x', alpha=0.3)
        
     
        ax5 = fig.add_subplot(gs[1, 1:], projection='polar')
        
        categories_radar = ['Morphology', 'Syntax', 'Pragmatics']
        if self.semantic:
            categories_radar.insert(2, 'Semantics')

        morph_score = morph_df['sarcasm_ratio'].mean()
        syntax_score = (normal_len_mean - sarcasm_len_mean) / normal_len_mean * 100 if normal_len_mean > 0 else 0
        pragmatic_score = prag_df['difference'].mean()
        
        scores = [morph_score, syntax_score, pragmatic_score]
        if self.semantic:
            semantic_score = (stats['normal_similarity_mean'] - stats['sarcasm_similarity_mean']) * 100
            scores.insert(2, semantic_score)

        total_score = np.mean(scores)
        scores += scores[:1]  
        categories_radar.append('Overall')
        
        angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
        scores += scores[:1]  
        angles += angles[:1]
        
        ax5.plot(angles, scores, 'o-', linewidth=2, color='#ef4444', label='Sarcasm Identification Power')
        ax5.fill(angles, scores, alpha=0.25, color='#ef4444')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories_radar)
        ax5.set_ylim(0, 100)
        ax5.set_title('Integrated Analysis: Sarcasm Identification Power of Each Aspect', fontsize=14, fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax5.grid(True)
        
        plt.savefig('integrated_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    
        print("\n" + "="*70)
        print("📊 Comprehensive Report on Linguistic Analysis of Sarcasm")
        print("="*70)
        
        print(f"\n1️⃣  Morphological Analysis:")
        print(f"   • Sarcasm Identification Score: {morph_score:.1f}/100")
        print(f"   • Key Signal Words: {', '.join(morph_df.head(3)['morpheme'].tolist())}")
        
        print(f"\n2️⃣  Syntactic Analysis:")
        print(f"   • Sarcasm Identification Score: {syntax_score:.1f}/100")
        print(f"   • Sarcastic sentences are shorter by an average of {normal_len_mean - sarcasm_len_mean:.1f} tokens")

        if self.semantic:
            print(f"\n3️⃣  Semantic Analysis:")
            print(f"   • Sarcasm Identification Score: {semantic_score:.1f}/100")
            print(f"   • Semantic Similarity Difference: {stats['normal_similarity_mean'] - stats['sarcasm_similarity_mean']:.3f}")
            print(f"   • Polarity Mismatch: Sarcasm is larger by {stats['sarcasm_polarity_gap_mean'] - stats['normal_polarity_gap_mean']:.3f}")
        
        print(f"\n4️⃣  Pragmatic Analysis:")
        print(f"   • Sarcasm Identification Score: {pragmatic_score:.1f}/100")
        print(f"   • Strongest Signal: {prag_df.loc[prag_df['difference'].idxmax(), 'signal_en']}")
        
        print(f"\n🎯 Overall Sarcasm Identification Score: {total_score:.1f}/100")
        print("\n💡 Conclusion:")
        if self.semantic and semantic_score == max(scores[:-1]):
             print("   Semantic analysis is the most powerful indicator for identifying sarcasm!")
        print("   By integrating various linguistic aspects, we can understand the phenomenon of sarcasm from multiple perspectives.")
        print("="*70 + "\n")