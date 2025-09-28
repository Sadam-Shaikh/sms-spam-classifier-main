#!/bin/bash

# Create necessary directories
mkdir -p ~/.streamlit
mkdir -p /tmp/nltk_data

# Set NLTK data path
export NLTK_DATA=/tmp/nltk_data

# Install NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA')"
python -c "import nltk; nltk.download('stopwords', download_dir='$NLTK_DATA')"

# Verify NLTK data
echo "Verifying NLTK data..."
python -c "import nltk; nltk.data.path.append('$NLTK_DATA'); print('NLTK data path:', nltk.data.path); nltk.data.find('tokenizers/punkt'); nltk.data.find('corpora/stopwords')"