#!/bin/bash

# Create necessary directories
mkdir -p ~/.streamlit/
mkdir -p ~/nltk_data/tokenizers
mkdir -p ~/nltk_data/corpora

# Configure Streamlit
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

# Install NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', download_dir='~/nltk_data')"
python -c "import nltk; nltk.download('stopwords', download_dir='~/nltk_data')"

# Set NLTK data path
export NLTK_DATA=~/nltk_data