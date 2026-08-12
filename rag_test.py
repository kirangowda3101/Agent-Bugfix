import time
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

repo = Path("repos/django")

files = sorted(repo.glob("**/*.py"))
print(f"Files: {len(files)}")

docs = []
for f in files:
    try:
        text = f.read_text(errors="ignore")
    except Exception:
        continue
    docs.append((str(f.relative_to("repos/django")), text))

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

texts = []
metadatas = []
for path, text in docs:
    for chunk in splitter.split_text(text):
        texts.append(chunk)
        metadatas.append({"path": path})

print(f"Chunks: {len(texts)}")

print("Loading embedding model ...")
t0 = time.time()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(f"Model loaded in {time.time() - t0:.1f}s")

print("Building index ...")
t0 = time.time()
store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
build_time = time.time() - t0
print(f"Index built in {build_time:.1f}s")

query = "parse a duration string with negative values"
print(f"\nQuery: {query}\n")

results = store.similarity_search(query, k=5)
for i, r in enumerate(results, 1):
    print(f"{i}. {r.metadata['path']}")
    print("   ", r.page_content[:120].replace("\n", " "), "...")
    print()