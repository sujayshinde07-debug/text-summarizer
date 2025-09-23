import streamlit as st
from transformers import pipeline

# Load summarizer model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Streamlit UI
st.set_page_config(page_title="AI Text Summarizer", page_icon="📝", layout="wide")

st.title("📝 AI Text Summarizer")
st.write("Enter any text below and get a concise AI-generated summary.")

# Input text box
user_input = st.text_area("✍️ Paste your text here:", height=250)

# Summary length options
col1, col2, col3 = st.columns(3)
with col1:
    min_len = st.slider("Minimum length", 20, 100, 30)
with col2:
    max_len = st.slider("Maximum length", 50, 300, 100)
with col3:
    do_sample = st.checkbox("Creative Mode (sampling)", value=False)

# Summarize button
if st.button("🔍 Summarize"):
    if user_input.strip():
        with st.spinner("Generating summary... ⏳"):
            summary = summarizer(
                user_input,
                max_length=max_len,
                min_length=min_len,
                do_sample=do_sample
            )
        st.subheader("✅ Summary:")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("⚠️ Please enter some text to summarize.")
