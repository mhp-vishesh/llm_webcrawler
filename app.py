import streamlit as st
from utils import load_and_index_links, get_answer, load_llama_model  # Assuming these imported

st.set_page_config(page_title="Automotive Web Crawler", layout="wide")
st.title("🚗 Automotive Web Crawler POC")

# Load tokenizer and Llama model once and cache in Streamlit session state for reuse
if "tokenizer" not in st.session_state or "model" not in st.session_state:
    with st.spinner("Loading local Llama model (takes time)..."):
        st.session_state.tokenizer, st.session_state.model = load_llama_model()

# --------- Step 1: Build Knowledge Base ---------
st.header("Step 1: Build Knowledge Base from URLs")
urls_input = st.text_area(
    "Enter automotive URLs (one per line):",
    value="https://www.example.com\nhttps://www.another-example.com",
    height=150
)

if st.button("Build Knowledge Base"):
    urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
    if urls:
        with st.spinner("Fetching and indexing content..."):
            load_and_index_links(urls)
        st.success("✅ Knowledge base built and FAISS index saved.")
    else:
        st.warning("⚠️ Please enter at least one valid URL.")

# --------- Step 2: Ask Questions ---------
st.header("Step 2: Ask Questions")
query = st.text_input("Enter your automotive question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Retrieving context and querying LLM..."):
            answer = get_answer(query, st.session_state.tokenizer, st.session_state.model)
        st.subheader("Answer:")
        st.write(answer)
