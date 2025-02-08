def answer_question(model, preprocessor, book_text: str, question: str) -> str:
    # Preprocess book
    processed_book = preprocessor.preprocess_book(book_text)
    
    # Find most relevant chunk for the question
    chunks = processed_book['chunks']
    # Implement relevance scoring here
    
    # Get answer from model
    inputs = model.tokenizer(
        question,
        chunks[0],  # Use most relevant chunk
        return_tensors='pt',
        max_length=512,
        truncation=True
    )
    
    start_logits, end_logits = model(
        inputs['input_ids'],
        inputs['attention_mask']
    )
    
    # Convert logits to answer span
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits)
    
    answer = model.tokenizer.decode(
        inputs['input_ids'][0][answer_start:answer_end+1]
    )
    
    return answer 