import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack
import numpy as np

# Download stopwords only once
nltk.download('stopwords', quiet=True)

# Load trained models and components
@st.cache_resource
def load_models():
    model = joblib.load("streamlit_model.pkl")
    vectorizer = joblib.load("streamlit_vectorizer.pkl")
    scaler = joblib.load("streamlit_scaler.pkl")
    return model, vectorizer, scaler

model, vectorizer, scaler = load_models()

# Text cleaning function
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Feature estimation function
def estimate_features_from_text(text):
    """Estimate numerical features from text using enhanced heuristics"""
    text_lower = text.lower()
    
    # Start with neutral values
    trust_score = 5.0
    source_reputation = 5.0
    clickbait_score = 0.5
    plagiarism_score = 20.0
    
    # Strong indicators of legitimate news
    strong_credible_words = ['scientists', 'researchers', 'study', 'research', 'university', 'mit', 'harvard', 'stanford']
    academic_words = ['published', 'journal', 'nature', 'science', 'peer-reviewed', 'findings', 'data']
    official_words = ['official', 'government', 'agency', 'department', 'ministry', 'administration']
    
    # Strong indicators of fake news
    strong_fake_indicators = ['miracle cure', 'doctors hate', 'you won\'t believe', 'big pharma', 'government cover']
    clickbait_words = ['shocking', 'unbelievable', 'amazing', 'incredible', 'secret', 'exposed']
    
    # Boost trust for strong credible indicators
    for word in strong_credible_words:
        if word in text_lower:
            trust_score += 2.5
            source_reputation += 2.0
            clickbait_score -= 0.2
    
    for word in academic_words:
        if word in text_lower:
            trust_score += 2.0
            source_reputation += 1.5
            clickbait_score -= 0.1
    
    for word in official_words:
        if word in text_lower:
            trust_score += 1.5
            source_reputation += 1.0
    
    # Reduce trust for fake indicators
    for phrase in strong_fake_indicators:
        if phrase in text_lower:
            trust_score -= 3.0
            clickbait_score += 0.4
    
    for word in clickbait_words:
        if word in text_lower:
            trust_score -= 1.0
            clickbait_score += 0.3
    
    # Special handling for "breakthrough" - can be legitimate in scientific context
    if 'breakthrough' in text_lower:
        if any(word in text_lower for word in ['scientists', 'research', 'study', 'university']):
            trust_score += 1.0  # Legitimate scientific breakthrough
        else:
            trust_score -= 0.5  # Potentially clickbait
    
    # Technology and energy are often legitimate topics
    tech_words = ['technology', 'renewable', 'energy', 'solar', 'wind', 'innovation']
    for word in tech_words:
        if word in text_lower:
            trust_score += 0.5
            source_reputation += 0.3
    
    # Check for excessive punctuation (fake news indicator)
    if text.count('!') > 2 or text.count('?') > 2:
        trust_score -= 1.5
        clickbait_score += 0.2
    
    # Check for all caps (shouting/clickbait)
    caps_count = sum(1 for word in text.split() if word.isupper() and len(word) > 3)
    if caps_count > 0:
        trust_score -= caps_count * 0.5
        clickbait_score += caps_count * 0.1
    
    # Clamp values to reasonable ranges
    trust_score = max(0, min(10, trust_score))
    source_reputation = max(0, min(10, source_reputation))
    clickbait_score = max(0, min(1, clickbait_score))
    plagiarism_score = max(0, min(100, plagiarism_score))
    
    return [trust_score, source_reputation, clickbait_score, plagiarism_score]

# Enhanced prediction function
def predict_news(news_text):
    """Predict if news is fake or real with confidence score"""
    if not news_text.strip():
        return "Please enter some text to analyze."
    
    # Clean the text
    cleaned_text = clean_text(news_text)
    
    # Get text features
    text_features = vectorizer.transform([cleaned_text])
    
    # Estimate numerical features
    numerical_features = estimate_features_from_text(news_text)
    
    # Scale numerical features (suppress warnings)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numerical_features_scaled = scaler.transform([numerical_features])
    
    # Combine features
    combined_features = hstack([text_features, numerical_features_scaled])
    
    # Make prediction
    prediction = model.predict(combined_features)[0]
    probability = model.predict_proba(combined_features)[0]
    
    # Format result
    if prediction == 1:
        return f"✅ **Real News** (Confidence: {probability[1]:.1%})"
    else:
        return f"🛑 **Fake News** (Confidence: {probability[0]:.1%})"

# Streamlit app layout
st.title("📰 Enhanced Fake News Detection App")
st.write("Enter a news headline or article content to check if it's Real or Fake.")

# Add some information about the model
with st.expander("ℹ️ About this detector"):
    st.write("""
    This enhanced fake news detector uses:
    - **Text Analysis**: Examines the content and language patterns
    - **Credibility Scoring**: Analyzes trustworthiness indicators
    - **Clickbait Detection**: Identifies sensational language
    - **Source Reputation**: Estimates source reliability
    
    The model looks for patterns like:
    - Excessive use of sensational words
    - Clickbait phrases
    - Credible language patterns
    - Scientific or official terminology
    """)

# Initialize session state for user input
if 'example_text' not in st.session_state:
    st.session_state.example_text = ""

# Example texts for testing
st.subheader("📝 Try these examples:")
col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Real News Example"):
        st.session_state.example_text = "Scientists at MIT published a new study in Nature journal showing breakthrough in renewable energy technology"
        st.rerun()

with col2:
    if st.button("⚠️ Fake News Example"):
        st.session_state.example_text = "SHOCKING: Doctors HATE this miracle cure that Big Pharma doesn't want you to know!"
        st.rerun()

# Text input area
user_input = st.text_area(
    "Enter News Text:",
    value=st.session_state.example_text,
    placeholder="Paste your news headline or article here...",
    height=150
)

# Predict button
if st.button("🔍 Analyze News", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_news(user_input)
            
            # Display result with appropriate styling
            if "Real News" in result:
                st.success(f"**Analysis Result:** {result}")
            else:
                st.error(f"**Analysis Result:** {result}")
            
            # Show additional analysis
            st.subheader("📊 Detailed Analysis")
            
            # Estimate features for display
            features = estimate_features_from_text(user_input)
            trust_score, source_reputation, clickbait_score, plagiarism_score = features
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Trust Score", f"{trust_score:.1f}/10")
                st.metric("Source Reputation", f"{source_reputation:.1f}/10")
            with col2:
                st.metric("Clickbait Score", f"{clickbait_score:.1%}")
                st.metric("Content Quality", f"{100-plagiarism_score:.1f}%")
            
            # Show warning indicators
            text_lower = user_input.lower()
            warning_indicators = []
            
            if any(word in text_lower for word in ['shocking', 'unbelievable', 'miracle']):
                warning_indicators.append("🚨 Contains sensational language")
            if any(phrase in text_lower for phrase in ['you won\'t believe', 'doctors hate']):
                warning_indicators.append("🚨 Contains clickbait phrases")
            if any(word in text_lower for word in ['alien', 'ufo', 'conspiracy']):
                warning_indicators.append("🚨 Contains extreme claims")
            if user_input.count('!') > 2:
                warning_indicators.append("🚨 Excessive punctuation")
            
            if warning_indicators:
                st.subheader("⚠️ Warning Indicators")
                for indicator in warning_indicators:
                    st.write(indicator)
            
            # Show positive indicators
            positive_indicators = []
            if any(word in text_lower for word in ['study', 'research', 'published']):
                positive_indicators.append("✅ Contains research-related terms")
            if any(word in text_lower for word in ['according', 'official', 'report']):
                positive_indicators.append("✅ Contains official language")
            if any(word in text_lower for word in ['journal', 'university', 'institute']):
                positive_indicators.append("✅ References credible institutions")
            
            if positive_indicators:
                st.subheader("✅ Credibility Indicators")
                for indicator in positive_indicators:
                    st.write(indicator)

# Footer
st.markdown("---")
st.markdown("**Note:** This is an AI-powered tool for educational purposes. Always verify news from multiple reliable sources.")
