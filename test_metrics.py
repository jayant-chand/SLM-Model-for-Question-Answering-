from src.evaluation.metrics import QAEvaluator

def main():
    # Initialize the evaluator
    evaluator = QAEvaluator()
    
    # Test data
    predictions = [
        "The cat sat on the mat",
        "Paris is the capital of France",
        "The sky is blue"
    ]
    
    ground_truths = [
        "The cat sat on the mat",
        "Paris is the capital city of France",
        "The sky appears blue"
    ]
    
    # Evaluate
    results = evaluator.evaluate_predictions(predictions, ground_truths)
    
    # Print results
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 