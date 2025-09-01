# Embeddings con MEL (Hugging Face Transformers)
from transformers import AutoTokenizer, AutoModel
import torch, faiss, numpy as np
import requests, os
from PyPDF2 import PdfReader
import re

MODEL_ID = "IIC/MEL"  # encoder legal español
tok = AutoTokenizer.from_pretrained(MODEL_ID)
enc = AutoModel.from_pretrained(MODEL_ID)
device = torch.device('cuda')
enc = enc.to(device)

def embed(texts: list[str]) -> np.ndarray:
    embeddings = []
    with torch.no_grad():
        for text in texts:
            # Process one text at a time
            toks = tok([text], padding=True, truncation=True, return_tensors="pt").to(device)
            out = enc(**toks).last_hidden_state  # [1, T, H]
            # Mean pooling
            mask = toks.attention_mask.unsqueeze(-1)
            emb = (out * mask).sum(dim=1) / mask.sum(dim=1)
            # Normalize and add to list
            emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()
            embeddings.append(emb[0])  # Remove batch dimension
    return np.array(embeddings)

# Index FAISS
# Read PDF
with open('docs/ley_arrendamiento_urbanos.pdf', 'rb') as file:
    pdf = PdfReader(file)
    text = ' '.join(page.extract_text() for page in pdf.pages)

# Split into sections (assuming sections start with "Artículo" or similar)
sections = re.split(r'(?=(?:Artículo|TÍTULO|CAPÍTULO)\s+\d+)', text)
sections = [s.strip() for s in sections if s.strip()]

# Chunk long sections (if needed)
def chunk_text(text, max_length=250, overlap=50):
    if len(text) <= max_length:
        return [text]
    chunks = []
    sentences = text.split('.')
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) > max_length and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            # Keep last sentences for overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
            current_chunk = overlap_sentences
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    return chunks

DOCS_PATH = "data/chunks.tx"
if os.path.exists(DOCS_PATH):
    print("Loading preprocessed chunks...")
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
else:
    docs = []
    for i, section in enumerate(sections):
        print(f"Processing section {i+1}/{len(sections)}")
        chunks = chunk_text(section)
        print(f"Split into {len(chunks)} chunks")
        docs.extend(chunks)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.replace("\n", " ").strip() + "\n")

os.makedirs("data", exist_ok=True)
INDEX_PATH = "data/law_index.faiss"
EMBEDDINGS_PATH = "data/embeddings.npy"

if os.path.exists(INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
    print("Loading existing index and embeddings...")
    index = faiss.read_index(INDEX_PATH)
    E = np.load(EMBEDDINGS_PATH)
else:
    print("Generating embeddings...")
    E = embed(docs)
    print("Done! Generating Index...")
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    print("Saving index and embeddings...")
    faiss.write_index(index, INDEX_PATH)
    np.save(EMBEDDINGS_PATH, E)
    print("Done!")

def retrieve(query, k=10):
    q = embed([query])
    D, I = index.search(q, k)
    return [(docs[i], float(D[0][j])) for j,i in enumerate(I[0])]

# Generación con Ollama + deepseek-r1
def generate(prompt):
    resp = requests.post('http://localhost:11434/api/generate',
                        json={
                            "model": "mistral:7b",
                            "prompt": prompt,
                            "stream": False
                        })
    try:
        return resp.json()['response']
    except Exception as e:
        print(e)
        print(resp.text)
        raise e


def answer(question):
    hits = retrieve(question, k=10)
    context = "\n\n".join([f"[{j+1}] {t}" for j,(t,_) in enumerate(hits)])
    prompt = f"""Eres un asistente jurídico. Tienes un contexto generado por un LLM entrenado con textos jurídicos. Usa SOLO el CONTEXTO para responder en español,
citando [n] tras cada afirmación. Si falta contexto, di que no puedes responder.

PREGUNTA: {question}

CONTEXTO JURÍDICO:
{context}
"""
    response = generate(prompt)
    
    # Find all citation numbers in response, including ranges like [1, 3, 4, 9]
    citations = set()
    for match in re.findall(r'\[([^\]]+)\]', response):
        for n in re.split(r'[,\s]+', match):
            if n.isdigit():
                citations.add(int(n))
    if any(n > len(hits) or n < 1 for n in citations):
        response += "\n\nADVERTENCIA: La respuesta contiene referencias inválidas."
    
    if len(citations) != 0:
        response += "\n\nReferencias:\n"
        for cite in citations:
            response += f"[{cite}], {hits[cite-1][0]}\n"
    
    return response


if __name__ == '__main__':
    question = "He firmado un contrato de alquiler de una vivienda por 1 año. ¿Por cuánto tiempo puedo renovar mi contrato de alquiler?"
    print(answer(question))
    