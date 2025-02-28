#RAFT Training Data Generator
Generate training data for Retrieval Augmented Fine-Tuning (RAFT) from PDF documents using Streamlit and OpenAI's GPT models.
![image](https://github.com/user-attachments/assets/7212cebe-96ed-488c-838a-fd9a5ca7c533)

#About RAFT
RAFT (Retrieval Augmented Fine-Tuning) is a methodology for adapting language models to domain-specific Retrieval Augmented Generation (RAG). This app implements the RAFT data generation process to create high-quality training data from PDF documents.
#Features

PDF Processing: Convert PDFs to text using GPT-4o vision capabilities
Document Chunking: Split documents into meaningful segments for question generation
Training Data Generation: Create question-answer pairs with oracle and distractor documents
Customizable Parameters: Adjust all aspects of the RAFT process
Dataset Export: Download your datasets in JSONL format ready for fine-tuning

#Installation
Clone this repository:
`git clone https://github.com/yourusername/raft-datagen.git`
`cd raft-datagen`

Create and activate a virtual environment:
`python -m venv venv`
`source venv/bin/activate  # On Windows: venv\Scripts\activate`

Install the required dependencies:
`pip install -r requirements.txt`

For PDF to image conversion, you'll need to install poppler:

`On Linux: sudo apt-get install -y poppler-utils`
`On Mac: brew install poppler`
`On Windows: conda install -c conda-forge poppler`

Usage
Run the Streamlit app:
`streamlit run streamlit_app.py`

The app will open in your browser.

1) Enter your OpenAI API key
2) Upload a PDF document
3) Configure RAFT parameters
4) Click "Process PDF and Generate Training Data"
5) Download the generated datasets
