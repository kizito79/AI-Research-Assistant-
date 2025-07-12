import streamlit as st
import fitz  # PyMuPDF
from transformers import BloomTokenizerFast, BloomForCausalLM
import requests

# === Load BLOOM model once ===
@st.cache_resource
def load_model():
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
    return tokenizer, model

tokenizer, model = load_model()

# === Tavily API key (Insert yours here) ===
TAVILY_API_KEY = "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"

# === Summarize text using BLOOM ===
def summarize_text(text, max_tokens=200):
    inputs = tokenizer(text[:1000], return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === Read PDF file ===
def read_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === Format APA citation ===
def format_apa(author, year, title, journal, volume, issue, pages, doi):
    return f"{author} ({year}). {title}. *{journal}*, {volume}({issue}), {pages}. https://doi.org/{doi}"

# === Tavily Web Search ===
def search_with_tavily(query):
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {"query": query, "search_depth": "advanced"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        return [{"title": "Error", "url": str(e)}]

# === Streamlit UI ===
st.set_page_config(page_title="Research Writing AI", layout="centered")
st.title("📄 Research Writing AI Assistant")
st.markdown("Summarize text, extract from PDFs, generate citations, and search the web in real-time!")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Summarize Text", "📥 Upload PDF", "📚 APA Citation", "🔍 Real-Time Research"])

# === TAB 1: Text Input ===
with tab1:
    st.subheader("🧠 Enter Research Text")
    user_input = st.text_area("Paste research content to summarize")
    if st.button("Generate Summary", key="text_summary"):
        if user_input.strip():
            with st.spinner("Summarizing..."):
                st.success(summarize_text(user_input))
        else:
            st.warning("Enter some text to summarize.")

# === TAB 2: PDF Upload ===
with tab2:
    st.subheader("📄 Upload a PDF File")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf and st.button("Summarize PDF"):
        with st.spinner("Processing..."):
            pdf_text = read_pdf(uploaded_pdf)
            if pdf_text.strip():
                st.success(summarize_text(pdf_text))
            else:
                st.error("No readable text found in PDF.")

# === TAB 3: Citation Generator ===
with tab3:
    st.subheader("📚 APA Citation Generator")
    with st.form("citation_form"):
        col1, col2 = st.columns(2)
        with col1:
            author = st.text_input("Author(s)", placeholder="e.g., Yusuf, A.")
            year = st.text_input("Year", placeholder="e.g., 2023")
            title = st.text_input("Title of Paper")
        with col2:
            journal = st.text_input("Journal Name")
            volume = st.text_input("Volume", placeholder="e.g., 12")
            issue = st.text_input("Issue", placeholder="e.g., 4")
            pages = st.text_input("Pages", placeholder="e.g., 101–110")
            doi = st.text_input("DOI", placeholder="e.g., 10.1234/journal.2023.104")
        submitted = st.form_submit_button("Generate Citation")
        if submitted:
            if all([author, year, title, journal, volume, issue, pages, doi]):
                st.code(format_apa(author, year, title, journal, volume, issue, pages, doi), language="markdown")
            else:
                st.warning("All fields are required.")

# === TAB 4: Real-Time Research with Tavily ===
with tab4:
    st.subheader("🔍 Live Web Research")
    search_query = st.text_input("Enter your research query (e.g., impact of urbanization on groundwater)")
    if st.button("Search Now"):
        if not TAVILY_API_KEY or "your_tavily_api_key_here" in TAVILY_API_KEY:
            st.error("Please add a valid Tavily API key in the script.")
        elif search_query.strip():
            with st.spinner("Searching the web..."):
                results = search_with_tavily(search_query)
                if results:
                    st.success("Top Results with Citations:")
                    for r in results[:5]:
                        st.markdown(f"**{r['title']}**  \n[{r['url']}]({r['url']})")
                        if 'content' in r:
                            st.caption(r['content'][:200] + "...")
                else:
                    st.info("No results found.")
        else:
            st.warning("Enter a search term.")