import json
import os

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as e:
    print(f"Missing package: {e.name}")
    print("Install with: pip install chromadb sentence-transformers")
    exit(1)

# Paths
KNOWLEDGE_BASE_PATH = "data/plant_knowledge_base.json"
CHROMA_DIR = "data/chroma_db"

def main():
    # Load knowledge base
    print("Loading knowledge base...")
    with open(KNOWLEDGE_BASE_PATH, encoding='utf-8') as f:
        knowledge_base = json.load(f)
    print(f"Loaded {len(knowledge_base)} plant entries")

    # Load embedding model
    print("\nLoading embedding model (sentence-transformers)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded")

    # Set up ChromaDB
    print(f"\nSetting up ChromaDB at {CHROMA_DIR}...")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection("medicinal_plants")
        print("Deleted existing collection")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name="medicinal_plants",
        metadata={"hnsw:space": "cosine"}
    )
    print("Created collection: medicinal_plants")

    # Ingest each plant
    print("\nIngesting plants into ChromaDB...")
    documents = []
    embeddings = []
    ids = []
    metadatas = []

    for i, plant in enumerate(knowledge_base):
        latin_name = plant['latin_name']
        common_name = plant['common_name']
        text = plant['text']

        # Create embedding
        embedding = model.encode(text).tolist()

        documents.append(text)
        embeddings.append(embedding)
        ids.append(latin_name)
        metadatas.append({
            "latin_name": latin_name,
            "common_name": common_name
        })

        print(f"  [{i+1}/{len(knowledge_base)}] Embedded: {common_name}")

    # Add all to ChromaDB
    print("\nAdding to ChromaDB...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"Total plants ingested: {collection.count()}")
    print(f"ChromaDB saved at: {CHROMA_DIR}")

    # Quick test query
    print(f"\n--- Quick Test ---")
    test_query = "Tulsi medicinal uses Nepal"
    test_embedding = model.encode(test_query).tolist()
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=2
    )
    print(f"Test query: '{test_query}'")
    print(f"Top result: {results['metadatas'][0][0]['common_name']}")
    print(f"Preview: {results['documents'][0][0][:200]}...")
    print(f"\nChromaDB is working correctly!")

if __name__ == "__main__":
    main()
