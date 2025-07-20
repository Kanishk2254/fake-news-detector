# 📰 Fake News Detector

An intelligent machine learning application that analyzes news articles and headlines to determine their authenticity. This project combines natural language processing, sentiment analysis, and heuristic-based scoring to identify potentially fake or misleading news content.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.0+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

### Core Functionality
- **Real-time Analysis**: Instant fake news detection for headlines and articles
- **Multi-factor Assessment**: Combines text analysis with credibility indicators
- **Confidence Scoring**: Provides probability scores for predictions
- **Interactive Web Interface**: User-friendly Streamlit-based application

### Advanced Analysis
- **Trust Score Calculation**: Evaluates content trustworthiness (0-10 scale)
- **Source Reputation Assessment**: Analyzes language patterns indicating reliable sources
- **Clickbait Detection**: Identifies sensational and misleading language
- **Content Quality Metrics**: Evaluates overall article quality

### Detection Capabilities
- ✅ **Credible Content Indicators**:
  - Scientific and research terminology
  - Official language patterns
  - Academic and institutional references
  - Proper news formatting

- 🚨 **Fake News Indicators**:
  - Sensational language and clickbait phrases
  - Excessive punctuation and capitalization
  - Conspiracy theories and extreme claims
  - Emotional manipulation techniques

## 🛠️ Technology Stack

- **Machine Learning**: Scikit-learn (Logistic Regression with balanced classes)
- **Natural Language Processing**: NLTK, TF-IDF Vectorization
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Feature Scaling**: StandardScaler for numerical features
- **Text Processing**: Custom cleaning and stop word removal

## 📊 Model Architecture

The fake news detector uses a hybrid approach combining:

1. **Text Features**: TF-IDF vectorization with n-grams (1-2)
2. **Numerical Features**: Heuristic-based scoring system
   - Trust Score (0-10)
   - Source Reputation (0-10)
   - Clickbait Score (0-1)
   - Content Quality Score (0-100)

3. **Balanced Classification**: Logistic Regression with class weights to handle imbalanced data

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn nltk scipy joblib
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Kanishk2254/fake-news-detector.git
cd fake-news-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run balanced_app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

### Web Application
1. **Enter Text**: Paste a news headline or article content
2. **Click Analyze**: Press the "🔍 Analyze News" button
3. **View Results**: Get instant classification with confidence scores
4. **Detailed Analysis**: Review trust metrics and warning indicators

### Example Inputs
**Real News Example**:
```
Scientists at MIT published a new study in Nature journal showing breakthrough in renewable energy technology
```

**Fake News Example**:
```
SHOCKING: Doctors HATE this miracle cure that Big Pharma doesn't want you to know!
```

## 📁 Project Structure

```
fake-news-detector/
├── balanced_app.py              # Streamlit web application
├── balanced_trainer.py          # Model training script
├── fake_news_dataset.csv        # Training dataset
├── streamlit_model.pkl          # Trained model for web app
├── streamlit_vectorizer.pkl     # TF-IDF vectorizer
├── streamlit_scaler.pkl         # Feature scaler
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

## 🔧 Model Training

To retrain the model with your own data:

1. **Prepare Dataset**: Ensure your CSV has columns: `title`, `label` ('Real'/'Fake'), and numerical features
2. **Run Training**:
```bash
python balanced_trainer.py
```
3. **Model Artifacts**: The script generates model files for the Streamlit app

### Training Features
- **Balanced Classes**: Uses class weights to handle data imbalance
- **Cross-validation**: Stratified train-test split
- **Feature Engineering**: Combines text and numerical features
- **Performance Metrics**: Detailed accuracy and classification reports

## 📈 Performance

The model achieves balanced performance across both fake and real news categories:
- **Balanced Classification**: Uses class weights to prevent bias
- **Multi-feature Analysis**: Combines text patterns with credibility indicators
- **Robust Detection**: Handles various types of misleading content

## 🎯 Use Cases

- **Journalism**: Quick verification tool for reporters and editors
- **Education**: Teaching media literacy and critical thinking
- **Social Media**: Personal fact-checking before sharing content
- **Research**: Analysis of misinformation patterns and trends

## ⚠️ Important Disclaimer

**This tool is designed for educational and assistance purposes only.**

- Always verify news from multiple reliable sources
- Use this as a supplementary tool, not the sole determinant
- Be aware that no automated system is 100% accurate
- Consider the source, context, and date of the information

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Areas
- Improve detection algorithms
- Add more language support
- Enhance UI/UX design
- Expand training dataset
- Add API endpoints

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NLTK** for natural language processing tools
- **Streamlit** for the amazing web framework
- **Scikit-learn** for machine learning algorithms
- **Open Source Community** for inspiration and resources

## 📞 Contact

**Developer**: Kanishk Shrama  
**GitHub**: [@Kanishk2254](https://github.com/Kanishk2254)  
**Project Link**: [https://github.com/Kanishk2254/fake-news-detector](https://github.com/Kanishk2254/fake-news-detector)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

*Together, we can fight misinformation and promote media literacy.*
