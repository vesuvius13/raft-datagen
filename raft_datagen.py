import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import base64
import tempfile
import re
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from openai import OpenAI
from pdf_processor import pdf_to_base64_urls, gpt_image_to_markdown
from text_processing import clean_chunk, strip_str, chunk_markdown_document
from data_generation import generate_instructions_gen, generate_label, add_chunk_to_dataset

# Initialize session state variables
def init_session_state():
    if 'train_df' not in st.session_state:
        st.session_state.train_df = None
    if 'val_df' not in st.session_state:
        st.session_state.val_df = None
    if 'test_df' not in st.session_state:
        st.session_state.test_df = None
    if 'generated' not in st.session_state:
        st.session_state.generated = False
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'markdown_doc' not in st.session_state:
        st.session_state.markdown_doc = ""
    if 'processing_progress' not in st.session_state:
        st.session_state.processing_progress = 0
    if 'current_processing_step' not in st.session_state:
        st.session_state.current_processing_step = ""
    if 'previous_upload_state' not in st.session_state:
        st.session_state.previous_upload_state = False

def reset_session_state():
    """Reset all relevant session state variables"""
    st.session_state.train_df = None
    st.session_state.val_df = None
    st.session_state.test_df = None
    st.session_state.generated = False
    st.session_state.chunks = []
    st.session_state.markdown_doc = ""
    st.session_state.processing_progress = 0
    st.session_state.current_processing_step = ""

def create_jsonl_content(df: pd.DataFrame, is_chat_format: bool = True) -> str:
    """
    Convert DataFrame to JSONL string content
    """
    jsonl_content = []
    
    if is_chat_format:
        # Chat format for fine-tuning
        for _, row in df.iterrows():
            entry = {
                "messages": [
                    {"role": "user", "content": row['instruction']},
                    {"role": "assistant", "content": row['cot_answer']}
                ]
            }
            jsonl_content.append(json.dumps(entry, ensure_ascii=False))
    else:
        # Standard format
        for _, row in df.iterrows():
            jsonl_content.append(json.dumps(row.to_dict(), ensure_ascii=False))
    
    return '\n'.join(jsonl_content)

def main():
    init_session_state()
    
    st.title("LLM Dataset Generator")
    st.markdown("[**RAFT: Adapting Language Model to Domain Specific RAG**](https://gorilla.cs.berkeley.edu/blogs/9_raft.html)", unsafe_allow_html=True)
    st.write("""
    Generate training/validation/test data for Retrieval Augmented Fine-Tuning (RAFT) from PDF documents.
    This app follows the process described in the RAFT methodology to create chain-of-thought question-answer pairs with relevant context.
    """)

    # Sidebar configurations
    st.sidebar.header("Configuration")
    
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    model_options = {
        "GPT-4o": "gpt-4o", 
        "GPT-4o-mini": "gpt-4o-mini",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo"
    }
    
    vision_model = st.sidebar.selectbox(
        "Select Vision Model (for PDF to Markdown)",
        ["GPT-4o", "GPT-4o-mini"]
    )
    
    qa_model = st.sidebar.selectbox(
        "Select Model (for QA Generation)",
        ["GPT-4o", "GPT-4", "GPT-4o-mini", "GPT-3.5 Turbo"]
    )
    
    # RAFT parameters
    st.sidebar.header("RAFT Parameters")
    
    questions_per_chunk = st.sidebar.number_input(
        "Questions per Chunk",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of questions to generate for each document chunk"
    )
    
    distractor_docs = st.sidebar.number_input(
        "Distractor Documents",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of distractor documents to include with each question. Distractor documents are irrelevant chunks that don't contain the answer but are included to teach the model to distinguish between relevant and irrelevant information."
    )
    
    oracle_prob = st.sidebar.slider(
        "Oracle Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Probability of including the oracle (original) document in the context. The oracle document is the one that contains the information needed to answer the question. Setting this below 1.0 helps train the model to recognize when it doesn't have enough information to answer."
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Creativity level for question generation"
    )
    
    # Adding system prompt for question generation
    question_system_prompt = st.sidebar.text_area(
        "System Prompt for Questions",
        value="""You are a synthetic question-answer pair generator. 
Given a chunk of context about some topic(s), generate example questions a user could ask 
and would be answered using information from the chunk. 
Do not generate questions about page numbers, section headings, or document structure.""",
        help="System prompt for generating questions from document chunks"
    )
    
    answer_system_prompt = st.sidebar.text_area(
        "System Prompt for Answers",
        value="You are a helpful question answerer who can provide an answer given a question and relevant context.",
        help="System prompt for generating answers to questions"
    )

    # Dataset split configuration
    st.sidebar.header("Dataset Split")
    
    train_size = st.sidebar.slider(
        "Training Set Size (%)",
        min_value=50,
        max_value=90,
        value=80,
        step=5
    )
    
    test_size = st.sidebar.slider(
        "Test Set Size (%)",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )
    
    # Calculate validation size
    val_size = 100 - train_size - test_size
    st.sidebar.write(f"Validation Set Size: {val_size}%")

    # Output format configuration
    st.sidebar.header("Output Format")
    
    output_format = st.sidebar.selectbox(
        "Select Output Format",
        ["JSONL (Chat Format)", "JSONL (Full Data)", "CSV"]
    )

    # Main area
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Check if upload state has changed
    current_upload_state = uploaded_file is not None
    if current_upload_state != st.session_state.previous_upload_state:
        if not current_upload_state:  # File was removed
            reset_session_state()
        st.session_state.previous_upload_state = current_upload_state

    if uploaded_file is not None:
        if not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
            return

        # Process PDF button
        if st.button("Process PDF and Generate Training Data"):
            client = OpenAI(api_key=api_key)
            
            # Step 1: Convert PDF to base64 images
            with st.spinner("Converting PDF to images..."):
                image_data = pdf_to_base64_urls(uploaded_file)
                st.success(f"Converted {len(image_data)} pages to images")
            
            # Step 2: Convert images to markdown
            st.session_state.current_processing_step = "Converting images to Markdown"
            st.write(f"Step: {st.session_state.current_processing_step}")
            
            markdown_doc = ""
            progress_bar = st.progress(0)
            
            for i, img_data in enumerate(image_data):
                result = gpt_image_to_markdown(img_data, client, model_options[vision_model])
                markdown_doc += "\n" + result
                
                # Update progress
                progress = (i + 1) / len(image_data)
                progress_bar.progress(progress)
            
            st.session_state.markdown_doc = markdown_doc
            st.success("Successfully converted images to Markdown")
            
            # Step 3: Chunk the markdown document
            st.session_state.current_processing_step = "Splitting Markdown into chunks"
            st.write(f"Step: {st.session_state.current_processing_step}")
            
            with st.spinner("Chunking document..."):
                chunks = chunk_markdown_document(markdown_doc)
                st.session_state.chunks = chunks
                st.success(f"Created {len(chunks)} document chunks")
            
            # Step 4: Generate QA pairs for each chunk
            st.session_state.current_processing_step = "Generating question-answer pairs"
            st.write(f"Step: {st.session_state.current_processing_step}")
            
            all_data = pd.DataFrame()
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                df = add_chunk_to_dataset(
                    client,
                    chunks, 
                    chunk, 
                    questions_per_chunk, 
                    distractor_docs,
                    answer_system_prompt,
                    question_system_prompt,
                    model_options[qa_model],
                    temperature,
                    oracle_prob
                )
                
                if not df.empty:
                    all_data = pd.concat([all_data, df], ignore_index=True)
                
                # Update progress
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress)
                
                # Provide periodic updates
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    st.write(f"Processed {i + 1}/{len(chunks)} chunks, generated {len(all_data)} QA pairs so far")
            
            if not all_data.empty:
                # Step 5: Split into train/val/test sets
                st.session_state.current_processing_step = "Splitting into training/validation/test sets"
                st.write(f"Step: {st.session_state.current_processing_step}")
                
                # First split: train vs. (val+test)
                temp_train_size = train_size / 100
                temp_test_val_size = (test_size + val_size) / 100
                
                train_df, temp_df = train_test_split(
                    all_data,
                    train_size=temp_train_size,
                    random_state=42
                )
                
                # Second split: val vs. test from the remaining data
                test_ratio = test_size / (test_size + val_size)
                
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=test_ratio,
                    random_state=42
                )
                
                # Store in session state
                st.session_state.train_df = train_df
                st.session_state.val_df = val_df
                st.session_state.test_df = test_df
                st.session_state.generated = True
                
                st.success(f"Successfully generated and split data into {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test examples")

    # Display results if data has been generated
    if st.session_state.generated and st.session_state.train_df is not None:
        # Display column descriptions
        st.subheader("Dataset Column Descriptions")
        col_descriptions = {
            "id": "Unique identifier for the data point",
            "type": "Type of the data (typically 'general')",
            "question": "The generated question",
            "context": "Collection of documents including the oracle and distractor documents",
            "oracle_context": "The original document chunk that contains information to answer the question",
            "cot_answer": "Chain-of-thought answer generated from the oracle context",
            "instruction": "Formatted input that combines the context documents with the question"
        }
        
        desc_df = pd.DataFrame.from_dict(col_descriptions, orient='index', columns=['Description'])
        st.table(desc_df)
        
        # Display the dataframes
        if st.checkbox("Show Training Set Preview"):
            st.subheader("Training Set (First 5 rows)")
            st.dataframe(st.session_state.train_df.head())
        
        if st.checkbox("Show Validation Set Preview"):
            st.subheader("Validation Set (First 5 rows)")
            st.dataframe(st.session_state.val_df.head())
            
        if st.checkbox("Show Test Set Preview"):
            st.subheader("Test Set (First 5 rows)")
            st.dataframe(st.session_state.test_df.head())
        
        # Create download section
        st.subheader("Download Generated Datasets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Training Set")
            if output_format == "CSV":
                train_csv = st.session_state.train_df.to_csv(index=False)
                st.download_button(
                    label="Download Training Set (CSV)",
                    data=train_csv,
                    file_name="train_raft_data.csv",
                    mime="text/csv",
                    key="train_csv"
                )
            elif output_format == "JSONL (Chat Format)":
                train_jsonl = create_jsonl_content(st.session_state.train_df, is_chat_format=True)
                st.download_button(
                    label="Download Training Set (JSONL)",
                    data=train_jsonl,
                    file_name="train_raft_data.jsonl",
                    mime="application/jsonl",
                    key="train_jsonl_chat"
                )
            else:  # JSONL (Full Data)
                train_jsonl = create_jsonl_content(st.session_state.train_df, is_chat_format=False)
                st.download_button(
                    label="Download Training Set (JSONL)",
                    data=train_jsonl,
                    file_name="train_raft_data.jsonl",
                    mime="application/jsonl",
                    key="train_jsonl_full"
                )
        
        with col2:
            st.markdown("##### Validation Set")
            if output_format == "CSV":
                val_csv = st.session_state.val_df.to_csv(index=False)
                st.download_button(
                    label="Download Validation Set (CSV)",
                    data=val_csv,
                    file_name="val_raft_data.csv",
                    mime="text/csv",
                    key="val_csv"
                )
            elif output_format == "JSONL (Chat Format)":
                val_jsonl = create_jsonl_content(st.session_state.val_df, is_chat_format=True)
                st.download_button(
                    label="Download Validation Set (JSONL)",
                    data=val_jsonl,
                    file_name="val_raft_data.jsonl",
                    mime="application/jsonl",
                    key="val_jsonl_chat"
                )
            else:  # JSONL (Full Data)
                val_jsonl = create_jsonl_content(st.session_state.val_df, is_chat_format=False)
                st.download_button(
                    label="Download Validation Set (JSONL)",
                    data=val_jsonl,
                    file_name="val_raft_data.jsonl",
                    mime="application/jsonl",
                    key="val_jsonl_full"
                )
                
        with col3:
            st.markdown("##### Test Set")
            if output_format == "CSV":
                test_csv = st.session_state.test_df.to_csv(index=False)
                st.download_button(
                    label="Download Test Set (CSV)",
                    data=test_csv,
                    file_name="test_raft_data.csv",
                    mime="text/csv",
                    key="test_csv"
                )
            elif output_format == "JSONL (Chat Format)":
                test_jsonl = create_jsonl_content(st.session_state.test_df, is_chat_format=True)
                st.download_button(
                    label="Download Test Set (JSONL)",
                    data=test_jsonl,
                    file_name="test_raft_data.jsonl",
                    mime="application/jsonl",
                    key="test_jsonl_chat"
                )
            else:  # JSONL (Full Data)
                test_jsonl = create_jsonl_content(st.session_state.test_df, is_chat_format=False)
                st.download_button(
                    label="Download Test Set (JSONL)",
                    data=test_jsonl,
                    file_name="test_raft_data.jsonl",
                    mime="application/jsonl",
                    key="test_jsonl_full"
                )
        
        # Display statistics
        st.subheader("Dataset Statistics")
        
        total_examples = len(st.session_state.train_df) + len(st.session_state.val_df) + len(st.session_state.test_df)
        
        st.write(f"Total QA pairs: {total_examples}")
        st.write(f"Training set size: {len(st.session_state.train_df)} ({train_size}%)")
        st.write(f"Validation set size: {len(st.session_state.val_df)} ({val_size}%)")
        st.write(f"Test set size: {len(st.session_state.test_df)} ({test_size}%)")
        
        # Calculate additional stats if data exists
        if not st.session_state.train_df.empty:
            avg_q_len = st.session_state.train_df['question'].str.len().mean()
            avg_a_len = st.session_state.train_df['cot_answer'].str.len().mean()
            
            st.write(f"Average question length: {avg_q_len:.1f} characters")
            st.write(f"Average answer length: {avg_a_len:.1f} characters")

if __name__ == "__main__":
    st.set_page_config(
        page_title="RAFT Training Data Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()