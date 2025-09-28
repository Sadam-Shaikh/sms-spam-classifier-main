import os
import nltk
# Set NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

from flask import Flask, request, jsonify
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class TextPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers (keeping words with numbers)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenization
        words = nltk.word_tokenize(text)
        
        # Remove stopwords and short words (length < 2)
        words = [self.ps.stem(word) for word in words 
                if word not in self.stop_words 
                and len(word) > 1
                and not word.isdigit()]
        
        return ' '.join(words)

app = Flask(__name__)

# Initialize the preprocessor
preprocessor = TextPreprocessor()

# Load the improved model
try:
    with open('improved_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Improved model loaded successfully!")
    
    # Test the model with a known spam message
    test_message = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    cleaned_test = preprocessor.clean_text(test_message)
    test_pred = model.predict([cleaned_test])[0]
    print(f"Test prediction (should be 1 for SPAM): {test_pred}")
    
except Exception as e:
    print(f"Error loading improved model: {e}")
    model = None

def transform_text(text):
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email/SMS Spam Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            textarea {
                width: 100%;
                height: 150px;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                display: block;
                margin: 20px auto;
            }
            button:hover {
                background-color: #45a049;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                display: none;
            }
            .spam {
                background-color: #ffebee;
                color: #c62828;
                border: 1px solid #ef9a9a;
            }
            .ham {
                background-color: #e8f5e9;
                color: #2e7d32;
                border: 1px solid #a5d6a7;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Email/SMS Spam Classifier</h1>
            <p>Enter the text message you want to check for spam:</p>
            <textarea id="message" placeholder="Type your message here..."></textarea>
            <button onclick="predict()">Check for Spam</button>
            <div id="result"></div>
        </div>

        <script>
            function predict() {
                const message = document.getElementById('message').value;
                const resultDiv = document.getElementById('result');
                
                if (!message.trim()) {
                    alert('Please enter a message to check.');
                    return;
                }

                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    resultDiv.style.display = 'block';
                    if (data.prediction === 1) {
                        resultDiv.className = 'spam';
                        resultDiv.textContent = '⚠️ This message is SPAM!';
                    } else {
                        resultDiv.className = 'ham';
                        resultDiv.textContent = '✓ This message is NOT spam.';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    resultDiv.style.display = 'block';
                    resultDiv.className = 'error';
                    resultDiv.textContent = 'An error occurred. Please try again.';
                });
            }
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Clean and preprocess the message
        cleaned_message = preprocessor.clean_text(message)
        
        # Make prediction
        prediction = model.predict([cleaned_message])[0]
        probability = model.predict_proba([cleaned_message])[0]
        
        # Return the prediction (0 for not spam, 1 for spam)
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(max(probability)),
            'message': message,
            'cleaned_message': cleaned_message
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Download required NLTK data
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Run the Flask app
    app.run(debug=True, port=5000)
