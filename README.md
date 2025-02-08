# Book Question-Answering Small Language Model

A transformer-based Small Language Model (SLM) designed to answer questions about books with high accuracy and efficiency.

## Features

- Transformer-based architecture using DistilBERT
- Efficient text chunking and processing
- Support for long-form text analysis
- Multiple evaluation metrics (BLEU, ROUGE, Exact Match, F1)
- Checkpoint saving and model persistence
- Detailed logging and training metrics

## Project Structure

slm-book-qa/
├── data/
│   └── books/         # Place your book text files here
│       ├── book1.txt
│       └── book2.txt
├── src/
│   ├── preprocessing/ # Text preprocessing modules
│   │   ├── text_processor.py
│   │   └── data_preparation.py
│   ├── model/        # Model architecture
│   │   └── qa_model.py
│   ├── training/     # Training modules
│   │   └── trainer.py
│   ├── evaluation/   # Evaluation metrics
│   │   └── metrics.py
│   ├── utils/        # Helper functions
│   │   └── inference.py
│   └── train.py      # Main training script
├── tests/            # Unit tests
├── checkpoints/      # Saved model checkpoints
├── requirements.txt  # Project dependencies
└── README.md        # This documentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/slm-book-qa.git
cd slm-book-qa
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Place your book text files in the `data/books/` directory. Then prepare your question-answer pairs:

```python
from src.preprocessing.data_preparation import DataPreparator

# Initialize data preparator
preparator = DataPreparator("data/books")

# Load book
book_text = preparator.load_book("your_book.txt")

# Create QA pairs
qa_pairs = [
    {"question": "Who is the main character?", "answer": "John Smith"},
    {"question": "Where does the story take place?", "answer": "London"}
]

# Prepare dataset
dataset = preparator.create_qa_pairs(book_text, qa_pairs)
train_data, val_data = preparator.prepare_dataset(dataset)
```

### 2. Train the Model

```python
import torch
from src.model.qa_model import BookQAModel
from src.training.trainer import QATrainer

# Initialize model
model = BookQAModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup trainer
trainer = QATrainer(model, optimizer, device)

# Train model
for epoch in range(3):
    loss = trainer.train_epoch(train_loader, epoch)
    trainer.save_checkpoint(epoch, loss)
```

### 3. Make Predictions

```python
from src.utils.inference import answer_question
from src.preprocessing.text_processor import TextPreprocessor

preprocessor = TextPreprocessor()
question = "What happens in chapter 1?"
answer = answer_question(model, preprocessor, book_text, question)
print(f"Q: {question}\nA: {answer}")
```

## Model Architecture

The model uses a transformer-based architecture with the following components:

- **Encoder**: DistilBERT base (uncased)
- **QA Head**: Linear layer for answer span prediction
- **Input Processing**: 
  - Max sequence length: 512 tokens
  - Sliding window approach for long texts
  - Automatic text chunking

## Evaluation Metrics

The model's performance is evaluated using:

- **Exact Match Score**: Perfect match between prediction and ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **BLEU Score**: N-gram overlap metric
- **ROUGE Score**: Recall-oriented metric for generated text

## Training Parameters

Default hyperparameters:

- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Max sequence length: 512
- Optimizer: AdamW
- Weight decay: 0.01

## Example Output

```
Question: "Who is the main character in the book?"
Answer: "Elizabeth Bennet is the main character, a young woman known for her intelligence and wit."

Question: "What is the main theme of the story?"
Answer: "The main theme revolves around marriage, social class, and personal growth in 19th century England."
```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

1. Enter or paste your text passage in the context area
2. Type your question in the question input field
3. Click "Get Answer" to receive the response

### Command Line

Run the main script:
```bash
python main.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Hugging Face for the transformer models
- NLTK team for text processing tools
- PyTorch team for the deep learning framework

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/slm-book-qa