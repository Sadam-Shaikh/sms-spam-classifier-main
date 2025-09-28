import pandas as pd
import numpy as np
import pickle
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

# Load and preprocess the data
try:
    # Try reading with header first
    df = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Check if we have the expected columns
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    else:
        # If not, assume first column is label, second is message
        df = df.iloc[:, :2]
        df.columns = ['label', 'message']
        
except Exception as e:
    print(f"Error reading CSV with header: {e}")
    # Try without header
    try:
        df = pd.read_csv('spam.csv', header=None, encoding='latin-1')
        df = df.iloc[:, :2]
        df.columns = ['label', 'message']
    except Exception as e2:
        print(f"Error reading CSV without header: {e2}")
        exit(1)

# Map labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1, '0': 0, '1': 1, 0: 0, 1: 1})

# Initialize the preprocessor
preprocessor = TextPreprocessor()

# Apply text preprocessing
df['cleaned_text'] = df['message'].apply(preprocessor.clean_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# Create and train the model pipeline with better parameters
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,            # Minimum document frequency
        max_df=0.95,         # Maximum document frequency
        stop_words='english'
    )),
    ('classifier', MultinomialNB(
        alpha=0.1,  # Additive smoothing parameter
        fit_prior=True
    ))
])

print("Training the improved model...")
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("-" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the improved model
with open('improved_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nImproved model has been saved to 'improved_model.pkl'")

# Test the specific message that was misclassified
test_message = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
cleaned_message = preprocessor.clean_text(test_message)
prediction = model.predict([cleaned_message])[0]
probability = model.predict_proba([cleaned_message])[0]

print("\nTest message:", test_message)
print("Cleaned message:", cleaned_message)
print("Prediction:", "SPAM" if prediction == 1 else "NOT SPAM")
print(f"Confidence: {max(probability)*100:.2f}%")
print("Class probabilities:", {0: f"{probability[0]*100:.2f}% NOT SPAM", 1: f"{probability[1]*100:.2f}% SPAM"})
