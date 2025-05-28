# PDF Semantic Search with OpenAI & Pinecone

This project enables you to upload a PDF file, extract its text, generate embeddings using OpenAI's embedding API, store them in Pinecone, and perform semantic search and question-answering based on the content.

---

## 🔧 Features

- 📄 Extract text from PDF (using PyMuPDF)
- 🧱 Chunk text using token-based limits
- 🧠 Embed text using OpenAI (`text-embedding-ada-002`)
- 🧭 Vector storage and search with Pinecone
- ❓ Question answering using GPT-3.5-turbo with context

---

## 🧰 Requirements

- Python 3.8+
- Install dependencies:
  
```
pip install openai pinecone-client tiktoken PyMuPDF
````

---

## 🚀 How to Use

1. **Set your API keys:**

   * OpenAI: `client = OpenAI(api_key="your_openai_key")`
   * Pinecone: `pc = Pinecone(api_key="your_pinecone_key")`

2. **Set the path to your PDF file:**

   ```python
   pdf_path = r"C:\Path\To\Your\PDF.pdf"
   ```

3. **Run the script** and follow the logs:

   * Text extraction
   * Embedding
   * Upserting to Pinecone
   * Semantic search
   * Contextual question answering

---

## 📦 Project Structure

```
pdf_semantic_search/
│
├── main.py              # Main script
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🤖 Sample Question

```python
question = "Importance of AI?"
```

The system fetches the top matching chunks from Pinecone and asks GPT to generate an answer based on those.

---

## 📄 License

This project is MIT licensed. Feel free to use, modify, and build on it!

---

## 🙌 Acknowledgements

* [OpenAI](https://openai.com)
* [Pinecone](https://www.pinecone.io/)
* [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
* [tiktoken](https://github.com/openai/tiktoken)

````

---

### 📦 `requirements.txt`

openai
pinecone-client
tiktoken
PyMuPDF
````

---

