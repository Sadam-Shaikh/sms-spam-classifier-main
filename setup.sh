#!/bin/bash

# Exit on error
set -e

# Set Python version
export PYTHON_VERSION=3.9

# Create necessary directories
mkdir -p ~/.streamlit
NLTK_DATA=/tmp/nltk_data

# Set NLTK data path
export NLTK_DATA

# First install Python dependencies
echo "Installing Python $PYTHON_VERSION and dependencies..."

# Install Python 3.9 if not already installed
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 not found, installing..."
    sudo apt-get update
    sudo apt-get install -y python3.9 python3.9-venv
fi

# Create a virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel

# Install only the required dependencies
echo "Installing required packages..."
pip install -r requirements.txt

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
python3.9 setup_nltk.py

# Clean up
rm -f setup_nltk.py

# Create Streamlit config
cat > ~/.streamlit/config.toml << EOF
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
enableWebsocketCompression = false

[browser]
serverAddress = ""

[runner]
fixMatplotlib = false

[theme]
base = "light"
EOF

echo "\nSetup completed successfully!"
echo "Python version: $(python3.9 --version)"
echo "NLTK data is available at: $NLTK_DATA"
echo "Streamlit configuration created at: ~/.streamlit/config.toml"