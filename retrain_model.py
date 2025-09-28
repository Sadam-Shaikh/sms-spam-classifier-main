import pandas as pd
import numpy as np
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK components
ps = PorterStemmer()

# Text preprocessing function
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

# Drop any rows with missing values
df = df.dropna()

# Apply text preprocessing
df['transformed_text'] = df['message'].apply(transform_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['transformed_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42
)

# Create and train the model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

print("Training the model...")
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("-" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
with open('retrained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer separately
with open('retrained_vectorizer.pkl', 'wb') as f:
    pickle.dump(model.named_steps['tfidf'], f)

print("\nModel and vectorizer have been saved to 'retrained_model.pkl' and 'retrained_vectorizer.pkl'")
print("You can now use these files with the web application.")
