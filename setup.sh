#!/bin/bash

# Exit on error
set -e

# Create necessary directories
mkdir -p ~/.streamlit
NLTK_DATA=/tmp/nltk_data

# Set NLTK data path
export NLTK_DATA

# First install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-streamlit.txt

# Create a Python script for NLTK setup
cat > setup_nltk.py << 'EOF'
import os
import nltk
import sys

# Set NLTK data path
nltk_data_path = os.getenv('NLTK_DATA', '/tmp/nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

print(f"NLTK data path set to: {nltk_data_path}")

# Add to NLTK data path
nltk.data.path.append(nltk_data_path)
print(f"Current NLTK paths: {nltk.data.path}")

# Download required data
print("Downloading NLTK data...")
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=False)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=False)
    print("NLTK data downloaded successfully!")
except Exception as e:
    print(f"Error downloading NLTK data: {e}", file=sys.stderr)
    sys.exit(1)

# Verify downloads
print("\nVerifying NLTK data...")
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print("NLTK data verified successfully!")
except LookupError as e:
    print(f"NLTK data verification failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

# Run the setup script
echo "Running NLTK setup..."
python setup_nltk.py

# Clean up
rm -f setup_nltk.py

echo "\nSetup completed successfully!"
echo "NLTK data is available at: $NLTK_DATA"