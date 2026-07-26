import json
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text Preprocessing Function
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words
    ]
    return " ".join(cleaned_tokens)

# Load FAQs
@st.cache_data
def load_faqs():
    with open('faqs.json', 'r') as file:
        return json.load(file)

faqs = load_faqs()
questions = [faq["question"] for faq in faqs]
preprocessed_questions = [preprocess_text(q) for q in questions]

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

def get_best_match(user_query: str, threshold: float = 0.2):
    processed_query = preprocess_text(user_query)
    if not processed_query.strip():
        return "Please ask a specific question so I can assist you better."
    
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    if best_score < threshold:
        return "I'm sorry, I couldn't find a confident match for your question. Please try rephrasing or contact support."
    
    return faqs[best_idx]["answer"]

# Streamlit UI Setup
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 FAQ Assistance Chatbot")
st.write("Ask any questions about our services, shipping, or policies!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling
if prompt := st.chat_input("Ask a question..."):
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Bot Response
    bot_response = get_best_match(prompt)
    
    # Display and record Bot Response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})