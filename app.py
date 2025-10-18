# app.py
import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        font-weight: 600;
        font-size: 1.3rem;
    }
    .spam-prediction {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        border: 2px solid #dc2626;
    }
    .ham-prediction {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: 2px solid #059669;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    .info-box {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load the pre-trained model and vectorizer"""
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory as this script.")
        return None, None

def preprocess_text(text):
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing non-alphanumeric characters
    3. Tokenizing
    4. Removing stopwords and punctuation
    5. Applying stemming
    """
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    if not text or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove non-alphanumeric characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        # Fallback simple tokenization
        tokens = text.split()
    
    # Remove stopwords and punctuation, and apply stemming
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and token.isalpha():
            stemmed_token = stemmer.stem(token)
            processed_tokens.append(stemmed_token)
    
    # Join tokens back into a string
    return ' '.join(processed_tokens)

def predict_spam(message, vectorizer, model):
    """Predict if a message is spam or ham"""
    # Preprocess the message
    processed_message = preprocess_text(message)
    
    # Transform using the vectorizer
    features = vectorizer.transform([processed_message])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0]
        spam_prob = probability[1] if len(probability) > 1 else 0.0
        ham_prob = probability[0]
    else:
        spam_prob = 1.0 if prediction == 1 else 0.0
        ham_prob = 1.0 - spam_prob
    
    return {
        'prediction': 'spam' if prediction == 1 else 'ham',
        'spam_probability': spam_prob,
        'ham_probability': ham_prob,
        'confidence': max(spam_prob, ham_prob)
    }

# Main application
def main():
    # Header section
    st.markdown('<h1 class="main-header">📱 SMS Spam Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered spam detection for your messages</p>', unsafe_allow_html=True)
    
    # Load model and vectorizer
    vectorizer, model = load_artifacts()
    
    if vectorizer is None or model is None:
        st.stop()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        st.markdown("### ✍️ Enter Your SMS Message")
        message = st.text_area(
            "",
            height=150,
            placeholder="Type or paste your SMS message here...\n\nExample: 'Congratulations! You've won a $1000 gift card. Click here to claim your prize.'",
            help="Enter the SMS message you want to check for spam"
        )
        
        # Predict button
        predict_button = st.button("🚀 Analyze Message", use_container_width=True)
        
        # Prediction results
        if predict_button:
            if not message.strip():
                st.warning("⚠️ Please enter a message to analyze.")
            else:
                with st.spinner("🔍 Analyzing message..."):
                    result = predict_spam(message, vectorizer, model)
                    
                    # Display prediction
                    if result['prediction'] == 'spam':
                        st.markdown(
                            f'<div class="prediction-box spam-prediction">'
                            f'🚨 SPAM DETECTED<br>'
                            f'<small>Confidence: {result["confidence"]:.1%}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box ham-prediction">'
                            f'✅ NOT SPAM (Ham)<br>'
                            f'<small>Confidence: {result["confidence"]:.1%}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Probability metrics
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric(
                            "Spam Probability",
                            f"{result['spam_probability']:.1%}",
                            delta=None
                        )
                    with prob_col2:
                        st.metric(
                            "Ham Probability",
                            f"{result['ham_probability']:.1%}",
                            delta=None
                        )
    
    with col2:
        # Information sidebar
        st.markdown("### ℹ️ About This Tool")
        st.markdown("""
        <div class="info-box">
        This AI-powered classifier detects spam messages using machine learning. 
        It analyzes message content and patterns to identify potential spam.
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("### 🛡️ What We Detect")
        st.markdown("""
        - ❌ Prize & lottery scams
        - ❌ Urgent account alerts
        - ❌ Phishing attempts
        - ❌ Fake offers & deals
        - ❌ Suspicious links
        - ✅ Legitimate messages
        """)
        
        # Tips section
        st.markdown("### 💡 Safety Tips")
        st.markdown("""
        - Never share personal information
        - Don't click suspicious links
        - Verify unexpected messages
        - Use official channels
        - Report spam to your carrier
        """)
    
    # Examples section
    st.markdown("---")
    st.markdown("### 🧪 Try These Examples")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("🎯 Prize Scam", use_container_width=True):
            st.session_state.example = "Congratulations! You've won a $1000 Walmart gift card. Text YES to claim your prize now!"
    
    with example_col2:
        if st.button("🏦 Bank Alert", use_container_width=True):
            st.session_state.example = "URGENT: Your bank account has been suspended. Click here to verify your details immediately."
    
    with example_col3:
        if st.button("👋 Normal Message", use_container_width=True):
            st.session_state.example = "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you."
    
    # Set example text if selected
    if hasattr(st.session_state, 'example'):
        st.experimental_rerun()

# Initialize session state for examples
if 'example' not in st.session_state:
    st.session_state.example = ""

# Import pandas for NaN check (moved here to avoid circular import)
import pandas as pd

if __name__ == "__main__":
    main()
