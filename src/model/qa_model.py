import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class QAModel(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # QA output layers
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)  # 2 for start/end position
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get start and end logits
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
    
    def get_answer(self, start_logits, end_logits, input_ids, tokenizer):
        # Get most likely start and end positions
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        
        # Convert to Python integers
        start_idx = start_idx.item()
        end_idx = end_idx.item()
        
        # Decode tokens to get answer
        answer_tokens = input_ids[0][start_idx:end_idx + 1]
        answer = tokenizer.decode(answer_tokens)
        
        return answer

    def get_answer_with_score(self, start_logits, end_logits, input_ids, tokenizer):
        # Convert logits to probabilities
        start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
        end_probs = torch.nn.functional.softmax(end_logits, dim=-1)
        
        # Get the most likely start and end positions
        start_idx = torch.argmax(start_probs)
        end_idx = torch.argmax(end_probs)
        
        # Ensure end comes after start
        if end_idx < start_idx:
            end_idx = start_idx + 1
        
        # Get the confidence score (probability product)
        score = float(start_probs[0][start_idx] * end_probs[0][end_idx])
        
        # Convert token indices to text
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_idx:end_idx + 1])
        answer = tokenizer.convert_tokens_to_string(tokens)
        
        # Clean up the answer
        answer = answer.strip()
        answer = ' '.join(answer.split())
        
        return answer, score