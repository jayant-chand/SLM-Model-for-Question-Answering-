from typing import List, Dict, Tuple
import json
import re
from transformers import AutoTokenizer

class DataProcessor:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        return text
        
    def prepare_features(self, question: str, context: str) -> Dict:
        """Tokenize and create features for model input."""
        # Tokenize question and context
        encoded = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded.get('token_type_ids', None)
        }