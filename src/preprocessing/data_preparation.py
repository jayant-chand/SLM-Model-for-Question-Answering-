import json
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split

class DataPreparator:
    def __init__(self, books_dir: str):
        self.books_dir = books_dir
        
    def load_book(self, filename: str) -> str:
        """Load book content from file"""
        with open(f"{self.books_dir}/{filename}", 'r', encoding='utf-8') as f:
            return f.read()
            
    def create_qa_pairs(self, book_text: str, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Create training examples from book text and QA pairs
        qa_pairs format: [{"question": "...", "answer": "..."}]
        """
        processed_pairs = []
        for qa in qa_pairs:
            processed_pairs.append({
                "context": book_text,
                "question": qa["question"],
                "answer": qa["answer"],
                "answer_start": book_text.find(qa["answer"])  # Find answer position in text
            })
        return processed_pairs
    
    def prepare_dataset(self, data: List[Dict], test_size=0.2):
        """Split data into train and validation sets"""
        train_data, val_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=42
        )
        return train_data, val_data

# Usage example:
# preparator = DataPreparator("data/books")
# book_text = preparator.load_book("sample_book.txt")
# qa_pairs = [{"question": "Who is the main character?", "answer": "John Smith"}]
# dataset = preparator.create_qa_pairs(book_text, qa_pairs)
# train_data, val_data = preparator.prepare_dataset(dataset) 