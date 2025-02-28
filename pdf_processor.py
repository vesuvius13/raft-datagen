import streamlit as st
import io
import os
import base64
import tempfile
from typing import List
from pdf2image import convert_from_path
from openai import OpenAI

def pdf_to_base64_urls(uploaded_file) -> List[str]:
    """
    Converts each page of a PDF to a base64 encoded URL starting with 'data:image/jpeg'.
    """
    st.session_state.current_processing_step = "Converting PDF pages to base64 encoded images"
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        base64_urls = []
        
        progress_bar = st.progress(0)
        total_pages = len(images)
        
        for i, image in enumerate(images):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            img_byte_arr.seek(0)
            base64_encoded = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            base64_url = f"data:image/jpeg;base64,{base64_encoded}"
            base64_urls.append(base64_url)
            
            # Update progress
            progress = (i + 1) / total_pages
            progress_bar.progress(progress)
        
        return base64_urls
        
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return []
    finally:
        # Clean up the temporary file
        try:
            os.unlink(pdf_path)
        except Exception as e:
            st.warning(f"Could not delete temporary file: {str(e)}")

def gpt_image_to_markdown(image_data: str, client: OpenAI, model: str) -> str:
    """
    Converts an image to markdown using GPT-4 vision.
    """
    messages = [
        {"role":"system", "content":"""You are an AI image assistant capable of extracting text from images.
         Given an image, you must extract any visible text on the image and return it in Markdown.
         You must keep the original layout and formatting of the text as much as possible in Markdown format.
         Pay attention to the text size and use headers, subheaders, bold, italic, tables etc where necessary."""},
         {"role":"user", "content":[{
                "type":"image_url",
                "image_url":{
                    "url":image_data
                    }
         }]}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error converting image to markdown: {str(e)}")
        return ""