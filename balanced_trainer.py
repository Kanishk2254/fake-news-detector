import pandas as pd
import numpy as np
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Load data
print("Loading dataset...")
df = pd.read_csv("fake_news_dataset.csv")
df.dropna(inplace=True)

print(f"Dataset loaded with {len(df)} samples")
print(f"Label distribution: {df['label'].value_counts()}")

# Convert labels to numeric (1 for Real, 0 for Fake)
df['label_numeric'] = df['label'].map({'Real': 1, 'Fake': 0})

# Since text content is too uniform, let's focus on numerical features
# that actually distinguish between fake and real news
print("\n=== ANALYZING NUMERICAL FEATURES ===")

numerical_features = [
    'sentiment_score', 'word_count', 'char_count', 'readability_score',
    'num_shares', 'num_comments', 'trust_score', 'source_reputation',
    'clickbait_score', 'plagiarism_score'
]

# Check correlation of numerical features with labels
print("Feature correlations with label:")
for feature in numerical_features:
    correlation = df[feature].corr(df['label_numeric'])
    print(f"  {feature}: {correlation:.4f}")

# Create a more balanced approach using weighted features
print("\n=== CREATING BALANCED MODEL ===")

# Use the most discriminative numerical features
key_features = ['trust_score', 'source_reputation', 'clickbait_score', 'plagiarism_score']
X_numerical = df[key_features]

# Still include some text features but with less weight
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

df['clean_title'] = df['title'].apply(clean_text)

# Create a simple text vectorizer with limited features
text_vectorizer = TfidfVectorizer(max_features=50, min_df=5, ngram_range=(1, 2))
X_text = text_vectorizer.fit_transform(df['clean_title'])

# Scale numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Combine features
from scipy.sparse import hstack
X_combined = hstack([X_text, X_numerical_scaled])

# Split data
y = df['label_numeric']
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train with adjusted parameters to avoid bias
model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    C=1.0,  # Regularization parameter
    max_iter=1000
)

print("\nTraining balanced model...")
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Detailed evaluation
y_pred = model.predict(X_test)
print(f"\nTest Set Evaluation:")
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Check prediction distribution
print(f"\nPrediction Distribution on Test Set:")
unique, counts = np.unique(y_pred, return_counts=True)
for cls, count in zip(unique, counts):
    label = "Real" if cls == 1 else "Fake"
    print(f"  {label}: {count} samples ({count/len(y_pred)*100:.1f}%)")

# Save the balanced model
print("\nSaving balanced model...")
joblib.dump(model, "balanced_model.pkl")
joblib.dump(text_vectorizer, "balanced_vectorizer.pkl")
joblib.dump(scaler, "balanced_scaler.pkl")

# Create a prediction function for the streamlit app
def create_streamlit_compatible_model():
    """Create a model that works with text-only input for Streamlit"""
    print("\nCreating Streamlit-compatible model...")
    
    # For Streamlit, we need to simulate the numerical features
    # since users will only input text
    
    # Create a rule-based system to estimate numerical features from text
    def estimate_features_from_text(text):
        text_lower = text.lower()
        
        # Simple heuristics based on common patterns
        trust_score = 5.0  # Default neutral
        source_reputation = 5.0  # Default neutral
        clickbait_score = 0.5  # Default neutral
        plagiarism_score = 20.0  # Default neutral
        
        # Adjust based on text content
        suspicious_words = ['breaking', 'shocking', 'unbelievable', 'miracle', 'secret']
        clickbait_indicators = ['you won\'t believe', 'shocking truth', 'doctors hate']
        
        for word in suspicious_words:
            if word in text_lower:
                trust_score -= 1
                clickbait_score += 0.2
        
        for phrase in clickbait_indicators:
            if phrase in text_lower:
                trust_score -= 2
                clickbait_score += 0.3
        
        # News-like content gets higher trust
        if any(word in text_lower for word in ['study', 'research', 'report', 'according']):
            trust_score += 1
            source_reputation += 1
        
        # Extreme claims get lower trust
        if any(word in text_lower for word in ['alien', 'ufo', 'conspiracy', 'miracle cure']):
            trust_score -= 3
            clickbait_score += 0.4
        
        # Clamp values to reasonable ranges
        trust_score = max(0, min(10, trust_score))
        source_reputation = max(0, min(10, source_reputation))
        clickbait_score = max(0, min(1, clickbait_score))
        plagiarism_score = max(0, min(100, plagiarism_score))
        
        return [trust_score, source_reputation, clickbait_score, plagiarism_score]
    
    # Create wrapper prediction function
    def predict_with_text_only(text):
        # Clean text
        cleaned_text = clean_text(text)
        
        # Vectorize text
        text_features = text_vectorizer.transform([cleaned_text])
        
        # Estimate numerical features
        numerical_features = estimate_features_from_text(text)
        numerical_features_scaled = scaler.transform([numerical_features])
        
        # Combine features
        combined_features = hstack([text_features, numerical_features_scaled])
        
        # Make prediction
        prediction = model.predict(combined_features)[0]
        probability = model.predict_proba(combined_features)[0]
        
        return prediction, probability
    
    return predict_with_text_only

# Create and save the Streamlit-compatible predictor
streamlit_predictor = create_streamlit_compatible_model()

# Save components needed for Streamlit
print("Saving components for Streamlit app...")
joblib.dump(model, "streamlit_model.pkl")
joblib.dump(text_vectorizer, "streamlit_vectorizer.pkl")
joblib.dump(scaler, "streamlit_scaler.pkl")

# Test the Streamlit predictor
print("\n=== TESTING STREAMLIT PREDICTOR ===")
test_samples = [
    "Scientists discover breakthrough in cancer research published in Nature journal",
    "President announces new economic policy to boost employment rates",
    "SHOCKING: Aliens land in New York City, government covers up evidence!",
    "Local weather forecast predicts sunny skies for the weekend",
    "YOU WON'T BELIEVE this miracle cure doctors don't want you to know!"
]

for i, sample in enumerate(test_samples, 1):
    pred, prob = streamlit_predictor(sample)
    label = "Real" if pred == 1 else "Fake"
    confidence = prob[1] if pred == 1 else prob[0]
    print(f"{i}. '{sample}'")
    print(f"   -> {label} (Confidence: {confidence:.4f})")
    print()

print("✅ Balanced model created successfully!")
print("Files saved:")
print("- balanced_model.pkl, balanced_vectorizer.pkl, balanced_scaler.pkl")
print("- streamlit_model.pkl, streamlit_vectorizer.pkl, streamlit_scaler.pkl")
