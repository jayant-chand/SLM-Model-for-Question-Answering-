import torch
from transformers import AutoTokenizer
from src.data.processor import DataProcessor
from src.model.qa_model import QAModel
from src.evaluation.metrics import QAEvaluator

def main():
    # Initialize components
    model_name = "bert-base-uncased"
    processor = DataProcessor(model_name)
    model = QAModel(model_name)
    evaluator = QAEvaluator()
    
    # Sample data
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
    Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists 
    and intellectuals for its design, but it has become a global cultural icon of France.
    """
    
    questions = [
        "Who designed the Eiffel Tower?",
        "When was the Eiffel Tower built?"
    ]
    
    ground_truths = [
        "Gustave Eiffel",
        "1887 to 1889"
    ]
    
    # Get predictions
    predictions = []
    model.eval()
    with torch.no_grad():
        for question in questions:
            # Prepare features
            features = processor.prepare_features(question, context)
            
            # Get model outputs
            start_logits, end_logits = model(**features)
            
            # Get answer
            answer = model.get_answer(
                start_logits, 
                end_logits, 
                features['input_ids'],
                processor.tokenizer
            )
            predictions.append(answer)
    
    # Evaluate
    results = evaluator.evaluate_predictions(predictions, ground_truths)
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()