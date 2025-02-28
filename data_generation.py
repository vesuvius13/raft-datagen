import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from text_processing import clean_chunk, strip_str

def generate_instructions_gen(client: OpenAI, chunk: str, x: int, model: str, temperature: float, system_prompt: str) -> List[str]:
    """
    Generates a list of questions based on an input document chunk
    """
    # Clean the chunk before processing
    cleaned_chunk = clean_chunk(chunk)
    
    # Format the system prompt to include the number of questions
    formatted_system_prompt = f"{system_prompt}\n\nGenerate exactly {x} example questions. Format: Return only the questions, one per line, without numbering or prefixes."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": cleaned_chunk}
        ],
        temperature=temperature
    )

    queries = response.choices[0].message.content.split('\n')
    queries = [strip_str(q) for q in queries]
    # Additional filtering to remove page/location reference questions
    queries = [q for q in queries if any(c.isalpha() for c in q) and
              not any(phrase in q.lower() for phrase in [
                  'page', 'section', 'chapter', 'where', 'located', 'find', 
                  'which part', 'document', 'paragraph'
              ])]
    queries = [q for q in queries if len(q.split()) >= 4 and '?' in q]
    return queries[:int(x)]

def generate_label(client: OpenAI, question: str, context: str, system_prompt: str, model: str) -> str:
    """
    Generates an Answer based on a Question-Document chunk pair
    """
    prompt = f"""
        Question: {question}\n Context: {context}\n
        Answer this question using the information given in the context above and no prior knowledge. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be given in a formal and technical tone.
        - If the answer cannot be found in the context, say "I'm sorry, I cannot answer this question as I'm missing the required information"
        You MUST begin your final answer with the tag "<ANSWER>:".
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

def add_chunk_to_dataset(
    client: OpenAI,
    chunks: List[str], 
    chunk: str, 
    x: int, 
    num_distract: int, 
    answer_system_prompt: str,
    question_system_prompt: str,
    model: str,
    temperature: float,
    p: float = 0.8
) -> pd.DataFrame:
    """
    Given a chunk, create {Q, A, D} triplets and add them to a dataframe.
    """
    i = chunks.index(chunk)
    cleaned_chunk = clean_chunk(chunk)
    
    try:
        qs = generate_instructions_gen(client, cleaned_chunk, x, model, temperature, question_system_prompt)
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return pd.DataFrame()
    
    data_points = []
    
    for q in qs:
        datapt = {
            "id": f"seed_task_{i}",
            "type": "general",
            "question": q,
            "oracle_context": chunk
        }

        # add num_distract distractor docs
        docs = [chunk]
        indices = list(range(0, len(chunks)))
        indices.remove(i)
        for j in np.random.choice(indices, num_distract, replace=False):
            docs.append(chunks[j])
        
        # decides whether to add oracle document
        oracle = np.random.uniform(0, 1) < p
        if not oracle:
            docs[0] = chunks[np.random.choice(indices, 1)[0]]
        np.random.shuffle(docs)

        d = {
            "title": [],
            "sentences": []
        }

        d["title"].append(["placeholder_title"]*(num_distract+1))
        d["sentences"].append(docs)
        datapt["context"] = d

        # add answer to q
        try:
            datapt["cot_answer"] = generate_label(client, q, cleaned_chunk, answer_system_prompt, model)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            continue

        # construct model instruction 
        context = ""
        for doc in docs:
            cleaned_doc = clean_chunk(doc)
            context += "<DOCUMENT>" + str(cleaned_doc) + "</DOCUMENT>\n"
        context += q
        datapt["instruction"] = context
        
        data_points.append(datapt)
    
    if data_points:
        return pd.DataFrame(data_points)
    else:
        return pd.DataFrame()