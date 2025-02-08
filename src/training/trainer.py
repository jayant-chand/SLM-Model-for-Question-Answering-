import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from pathlib import Path

class QATrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def save_checkpoint(self, epoch, loss, path="checkpoints"):
        Path(path).mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, f"{path}/checkpoint_epoch_{epoch}.pt")
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)
            
            # Forward pass
            start_logits, end_logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.calculate_loss(start_logits, end_logits, 
                                     start_positions, end_positions)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def calculate_loss(self, start_logits, end_logits, start_positions, end_positions):
        loss_fct = torch.nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        return (start_loss + end_loss) / 2 