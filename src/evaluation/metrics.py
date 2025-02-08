from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import nltk
from collections import Counter
import string
import re
from rouge_score import rouge_scorer

nltk.download('punkt')

class QAEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s):
        if not s: return []
        return self.normalize_answer(s).split()

    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(truth)) * 100

    def compute_f1(self, prediction, truth):
        pred_tokens = self.get_tokens(prediction)
        truth_tokens = self.get_tokens(truth)
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1 * 100

    def compute_rouge_l(self, prediction, truth):
        scores = self.scorer.score(prediction, truth)
        return scores['rougeL'].fmeasure * 100

    def evaluate(self, prediction, truth):
        return {
            'exact_match': self.compute_exact_match(prediction, truth),
            'f1': self.compute_f1(prediction, truth),
            'rouge_l': self.compute_rouge_l(prediction, truth)
        }

    def calculate_exact_match(self, prediction: str, ground_truth: str) -> float:
        return float(prediction.strip().lower() == ground_truth.strip().lower())
    
    def calculate_f1(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        common = set(pred_tokens) & set(truth_tokens)
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0
            
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0
            
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def calculate_bleu(self, prediction: str, ground_truth: str) -> float:
        return sentence_bleu([ground_truth.split()], prediction.split())
    
    def evaluate_predictions(self, predictions: List[str], 
                           ground_truths: List[str]) -> Dict:
        metrics = {
            'exact_match': [],
            'f1': [],
            'bleu': []
        }
        
        for pred, truth in zip(predictions, ground_truths):
            metrics['exact_match'].append(self.calculate_exact_match(pred, truth))
            metrics['f1'].append(self.calculate_f1(pred, truth))
            metrics['bleu'].append(self.calculate_bleu(pred, truth))
        
        # Calculate averages
        return {k: np.mean(v) for k, v in metrics.items()} 