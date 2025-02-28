import streamlit as st
from raft_datagen import main

# Set page configuration
st.set_page_config(
    page_title="RAFT Training Data Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run the main application
if __name__ == "__main__":
    main()