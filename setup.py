from setuptools import setup, find_packages

setup(
    name="raft-datagen",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.36.0",
        "openai>=1.55.0",
        "PyPDF2>=3.0.0",
        "scikit-learn>=1.0.0",
        "pdf2image>=1.17.0",
        "langchain_text_splitters>=0.0.1",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
)