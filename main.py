import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import logging
import PyPDF2
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self):
        self.model_name = "deepset/roberta-base-squad2"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_answer(self, question, context):
        try:
            # Tokenize input
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            # Remove overflow_to_sample_mapping from inputs if present
            if 'overflow_to_sample_mapping' in inputs:
                del inputs['overflow_to_sample_mapping']

            # Get model outputs
            outputs = self.model(**inputs)
            
            # Get start and end logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Get the most likely answer span
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)

            # Convert tokens to answer string
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens)

            # Calculate confidence score
            start_score = torch.softmax(start_logits, dim=1)[0][start_idx].item()
            end_score = torch.softmax(end_logits, dim=1)[0][end_idx].item()
            confidence = (start_score + end_score) / 2

            return answer, confidence

        except Exception as e:
            logger.error(f"Error in get_answer: {str(e)}")
            raise

def extract_text_from_pdf(pdf_file):
    try:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def main():
    st.title("Question Answering System")

    # Initialize QA system
    @st.cache_resource
    def load_qa_system():
        return QASystem()

    try:
        qa_system = load_qa_system()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Input method selection
    input_method = st.radio("Choose input method:", ["Text Input", "PDF Upload"])

    # Context input based on selected method
    if input_method == "Text Input":
        context = st.text_area("Enter your text:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
        if uploaded_file is not None:
            context = extract_text_from_pdf(uploaded_file)
            if context:
                st.success("PDF processed successfully!")
                with st.expander("Show extracted text"):
                    st.text(context[:1000] + "..." if len(context) > 1000 else context)
        else:
            context = None

    # Question input
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not context:
            st.warning("Please enter some text or upload a PDF first.")
            return
        if not question:
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Finding answer..."):
                answer, confidence = qa_system.get_answer(question, context)

                # Display results
                st.markdown("### Results")
                
                # Question and Answer columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Question:**")
                    st.info(question)
                
                with col2:
                    st.markdown("**Answer:**")
                    st.success(answer)

                # Confidence score
                st.markdown("**Confidence Score:**")
                st.progress(min(confidence, 1.0))
                st.caption(f"Confidence: {confidence:.2%}")

                # Store in history
                if 'qa_history' not in st.session_state:
                    st.session_state.qa_history = []
                
                st.session_state.qa_history.append({
                    'question': question,
                    'answer': answer,
                    'confidence': confidence
                })

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            logger.error(f"Error details: {str(e)}", exc_info=True)

    # Display history
    if 'qa_history' in st.session_state and st.session_state.qa_history:
        st.markdown("---")
        st.markdown("### Previous Questions & Answers")
        
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
            with st.expander(f"Q{len(st.session_state.qa_history)-i}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Question:**")
                    st.info(qa['question'])
                with col2:
                    st.markdown("**Answer:**")
                    st.success(qa['answer'])
                st.caption(f"Confidence: {qa['confidence']:.2%}")

if __name__ == "__main__":
    main()