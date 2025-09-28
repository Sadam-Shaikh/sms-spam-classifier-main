import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

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

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK components
ps = PorterStemmer()

# Load the model and vectorizer
try:
    with open('retrained_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('retrained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    exit(1)

# Test message
test_message = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"

# Transform the message
transformed_message = transform_text(test_message)
print("\nTransformed message:", transformed_message)

# Make prediction
vector_input = tfidf.transform([transformed_message])
prediction = model.predict(vector_input)[0]
probability = model.predict_proba(vector_input)[0]

print("\nPrediction:", "SPAM" if prediction == 1 else "NOT SPAM")
print(f"Confidence: {max(probability)*100:.2f}%")
print("Class probabilities:", {0: f"{probability[0]*100:.2f}% NOT SPAM", 1: f"{probability[1]*100:.2f}% SPAM"})
