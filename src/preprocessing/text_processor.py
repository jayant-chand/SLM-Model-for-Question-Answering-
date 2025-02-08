import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Dict

class TextPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        
    def preprocess_book(self, text: str) -> Dict:
        """
        Preprocess book text into chunks for easier processing
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Create chunks of text (e.g., 512 tokens each)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            if current_length + len(tokens) > 512:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(tokens)
            
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return {
            'chunks': chunks,
            'num_chunks': len(chunks)
        } 