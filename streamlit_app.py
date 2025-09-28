import streamlit as st
import pickle
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import sys

# Set page config
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide"
)

# Set NLTK data path
nltk_data_path = os.getenv('NLTK_DATA', '/tmp/nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Text preprocessing function
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    text = word_tokenize(text)
    # Remove special characters and keep only alphanumeric
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    # Join the words back into a string
    return " ".join(text)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

tfidf, model = load_model()

# Streamlit UI
st.title("📱 SMS Spam Classifier")
st.write("Enter an SMS message to check if it's spam or not.")

# Text input
message = st.text_area("Message", "")

if st.button("Check"):
    if message:
        # Preprocess the message
        transformed_message = transform_text(message)
        # Vectorize
        vector_input = tfidf.transform([transformed_message])
        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM")
        
        # Show confidence score
        proba = model.predict_proba(vector_input)[0]
        st.write(f"Confidence: {max(proba)*100:.2f}%")
    else:
        st.warning("Please enter a message to check.")

# Add some examples
expander = st.expander("Example messages")
with expander:
    st.write("Try these examples:")
    st.code("You've won a free prize! Click here to claim.")
    st.code("Hi John, just checking in to see how you're doing.")
    st.code("URGENT: Your account has been compromised. Click to secure now!")
    st.code("Meeting at 3 PM tomorrow. Don't forget your laptop.")
