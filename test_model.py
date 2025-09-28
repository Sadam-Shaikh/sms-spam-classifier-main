import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Import after downloading
from app_flask import transform_text

# Load the model and vectorizer with proper error handling
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    tfidf = None
    model = None

def predict_message(message):
    if tfidf is None or model is None:
        return "ERROR: Model not loaded"
    try:
        # Transform the message
        transformed_message = transform_text(message)
        # Vectorize
        vector_input = tfidf.transform([transformed_message])
        # Predict
        prediction = model.predict(vector_input)[0]
        return "SPAM" if prediction == 1 else "NOT SPAM"
    except Exception as e:
        return f"ERROR: {str(e)}"

# Test with some sample messages from the CSV
sample_messages = [
    ("You've won a free prize! Click here to claim.", "SPAM"),
    ("Hi John, just checking in to see how you're doing.", "NOT SPAM"),
    ("Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed £1000 cash or £5000 prize!", "SPAM"),
    ("Sorry, I'll call later", "NOT SPAM"),
    ("Your free ringtone is waiting to be collected. Simply text the password to 85069 to verify.", "SPAM")
]

print("Testing model with sample messages:")
print("-" * 60)
for message, expected in sample_messages:
    prediction = predict_message(message)
    print(f"Message: {message[:60]}...")
    print(f"Expected: {expected}")
    print(f"Predicted: {prediction}")
    print("-" * 60)

# Test with some messages from the CSV file
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    # Get some random samples (5 ham and 5 spam)
    ham_samples = df[df['v1'] == 'ham'].sample(3, random_state=42)
    spam_samples = df[df['v1'] == 'spam'].sample(2, random_state=42)
    test_samples = pd.concat([ham_samples, spam_samples])
    
    print("\nTesting model with sample messages from CSV:")
    print("-" * 80)
    for i, (_, row) in enumerate(test_samples.iterrows(), 1):
        message = row['v2'] if 'v2' in row else row[1]
        expected = "SPAM" if str(row['v1']).strip().lower() == 'spam' else "NOT SPAM"
        prediction = predict_message(message)
        
        print(f"\nMessage {i}:")
        print(f"Text: {str(message)[:100]}{'...' if len(str(message)) > 100 else ''}")
        print(f"Expected: {expected}")
        print(f"Predicted: {prediction}")
        print("-" * 80)
        
except Exception as e:
    print(f"Error reading CSV file: {e}")
    # Try alternative column names
    try:
        df = pd.read_csv('spam.csv', header=None, encoding='latin-1')
        print("\nTesting with alternative CSV format:")
        print("-" * 80)
        for i, (_, row) in enumerate(df.sample(5, random_state=42).iterrows(), 1):
            message = row[0] if len(row) > 0 else "No message"
            expected = "SPAM" if len(row) > 1 and str(row[1]).strip().lower() == 'spam' else "NOT SPAM"
            prediction = predict_message(message)
            
            print(f"\nMessage {i}:")
            print(f"Text: {str(message)[:100]}{'...' if len(str(message)) > 100 else ''}")
            print(f"Expected: {expected}")
            print(f"Predicted: {prediction}")
            print("-" * 80)
    except Exception as e2:
        print(f"Error with alternative CSV format: {e2}")
