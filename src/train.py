import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from model.qa_model import BookQAModel
from preprocessing.text_processor import TextPreprocessor

def train_model(train_data, val_data, epochs=3):
    model = BookQAModel()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_data:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            start_logits, end_logits = model(input_ids, attention_mask)
            
            loss = calculate_loss(start_logits, end_logits, start_positions, end_positions)
            loss.backward()
            optimizer.step()
            
    return model 