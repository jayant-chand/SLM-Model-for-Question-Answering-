from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import nltk

nltk.download('punkt')

class QAEvaluator:
    def __init__(self):
        pass  # Remove rouge initialization
        
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