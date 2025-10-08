import os
import requests
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import streamlit as st
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

INDEX_FILE = "faiss_index.pkl"

# --------- Step 1: Crawl URLs ---------
def fetch_text_from_url(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def load_and_index_links(urls):
    all_text = ""
    for url in urls:
        text = fetch_text_from_url(url)
        all_text += text + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(all_text)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face API token not found in environment variables as HF_TOKEN")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(docs, embeddings_model)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(db, f)

# --------- Step 2: Retrieve context ---------
def retrieve_context(query, k=3):
    if not os.path.exists(INDEX_FILE):
        return "⚠️ Build knowledge base first from links."

    with open(INDEX_FILE, "rb") as f:
        db = pickle.load(f)
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in results])
    return context

# --------- Step 3: Load local Llama model ---------
@st.cache_resource(show_spinner=False)
def load_llama_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return tokenizer, model

def query_local_llama(tokenizer, model, prompt, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False  # deterministic output
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# --------- Helper: Extract answer only ---------
def extract_answer_from_output(output_text):
    # Extract only content after "Answer:" token for display
    if "Answer:" in output_text:
        return output_text.split("Answer:", 1)[-1].strip()
    return output_text.strip()

# --------- Step 4: Combine retrieval + local LLM ---------
def get_answer(query, tokenizer, model):
    context = retrieve_context(query)
    if context.startswith("⚠️"):
        return context

    prompt = f"""
You are an expert in the German automotive industry. Using the latest information extracted from trusted sources such as the Volkswagen Group, VDA press releases, Autobild, Audi careers, and similar German automotive sector websites, provide a clear and concise summary of:

- The latest challenges and problems currently faced by the automotive industry in Germany.
- The newest trends and market developments.
- Emerging and innovative technologies gaining traction in this sector.

Answer based solely on the context below. If the information is not present, say 'Information not available in the provided context.'

Context:
{context}

Question:
{query}

Answer:
"""
    raw_output = query_local_llama(tokenizer, model, prompt)
    answer_only = extract_answer_from_output(raw_output)
    return answer_only

# --------- Streamlit UI ---------
def main():
    st.title("Local Llama-2 + FAISS Retrieval Demo")

    urls_input = st.text_area("Enter URLs to crawl (one per line)")
    if st.button("Build knowledge base"):
        urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
        if urls:
            load_and_index_links(urls)
            st.success("Knowledge base built and indexed successfully!")
        else:
            st.warning("Please enter at least one valid URL.")

    query = st.text_input("Ask a question")
    if query:
        tokenizer, model = load_llama_model()
        answer = get_answer(query, tokenizer, model)
        st.text_area("Answer", answer)

if __name__ == "__main__":
    main()
