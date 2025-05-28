import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# -------- 1️⃣ PDF TEXT EXTRACTION --------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# -------- 2️⃣ TEXT CHUNKING --------
def chunk_text(text, max_tokens=800):
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = tokenizer.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        decoded_chunk = tokenizer.decode(chunk)
        chunks.append(decoded_chunk)
    return chunks

# -------- 3️⃣ OPENAI EMBEDDING --------
client = OpenAI(api_key="----------------------")  # Replace with your key

def get_openai_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts  # must be a list of strings
    )
    return [res.embedding for res in response.data]

# -------- 4️⃣ PINECONE SETUP --------
pc = Pinecone(api_key="------------")  # Replace with your key

index_name = "pdf-embeddings"
embedding_dimension = 1536

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Index '{index_name}' created!")
else:
    print(f"ℹ️ Index '{index_name}' already exists.")

# Connect to index
index = pc.Index(index_name)

# -------- 5️⃣ PROCESS PDF --------
pdf_path = r"C:\Users\Gujjar\Desktop\AI.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
print("✅ PDF text extracted successfully!")

text_chunks = chunk_text(pdf_text)
print(f"✅ Split into {len(text_chunks)} chunks.")

embeddings = get_openai_embeddings(text_chunks)
print("✅ All chunks embedded successfully!")

# Upsert to Pinecone
vectors = [(f"chunk-{i}", embeddings[i], {"text": text_chunks[i]}) for i in range(len(embeddings))]
index.upsert(vectors=vectors)
print("✅ Embeddings stored in Pinecone!")

# -------- 6️⃣ SEARCH --------

query = "What is AI?"
query_embedding = get_openai_embeddings([query])[0]

search_result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
print("🔍 Search Results:")
for match in search_result.matches:
    print(f"- Score: {match.score:.2f}\n  Text: {match.metadata['text'][:200]}...\n")



# -------- 7️⃣ QUESTION ANSWERING --------

# Sample question (you can change this as needed)
question = " Importance of AI ?"

# Step 1: Get embedding for the question
question_embedding = get_openai_embeddings([question])[0]

# Step 2: Search Pinecone
search_response = index.query(vector=question_embedding, top_k=3, include_metadata=True)
top_chunks = [match.metadata['text'] for match in search_response.matches]

# Step 3: Combine references
context = "\n\n---\n\n".join(top_chunks)

# Step 4: Ask GPT to answer based on those chunks
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer the question using only the given context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
    ]
)

# Step 5: Print answer
print("🤖 GPT Answer:")
print(response.choices[0].message.content.strip())
