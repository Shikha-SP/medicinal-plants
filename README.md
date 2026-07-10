# 🌿 Nepal Medicinal Plant Intelligence System

> AI-powered medicinal plant identification for Nepal
---

## What It Does

Upload a photo of any plant found in Nepal and the system will:

- **Identify** the plant species using a DINOv2-powered image similarity search
- **Retrieve** medicinal information from a curated Wikipedia knowledge base
- **Generate** a structured report including medicinal uses, safety classification, traditional Nepali uses, and geographic distribution within Nepal

---

## System Architecture

```
User uploads plant photo
        ↓
Frontend (HTML/CSS/JS)
        ↓ POST /identify
FastAPI Backend
        ↓
DINOv2 + FAISS (plant identification)
        ↓ plant_name + confidence
ChromaDB RAG Pipeline
        ↓ Wikipedia knowledge chunks
Groq LLaMA 3.3 70B (report generation)
        ↓
JSON response → Frontend displays report
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | DINOv2 (Facebook) + FAISS similarity search |
| Dataset | iNaturalist Nepal observations (2450 plant species) |
| RAG | sentence-transformers + ChromaDB + Groq LLaMA 3.3 70B |
| Backend | FastAPI + Python |
| Frontend | HTML + CSS + JavaScript |
| Deployment | HuggingFace Spaces (Docker) |

---

## Project Structure

```
medicinal-plants/
├── main.py                          # FastAPI backend — entry point
├── predict.py                       # DINOv2 + FAISS plant identification
│
├── scripts/
│   ├── filter_medicinal_plants.py   # Filter 71 Nepal medicinal plants
│   ├── build_knowledge_base.py      # Wikipedia API scraper
│   └── ingest_chromadb.py           # Embed + store in ChromaDB
│
├── plant_pipeline/                  # ML pipeline (Sikha)
│   ├── vectorization.py             # Generate FAISS index from dataset
│   ├── query_index.py               # Search FAISS index
│   ├── search.py
│   ├── data_loader.py
│   └── cleanup.py
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── data/
│   ├── chroma_db/                   # ChromaDB vector database (generated locally)
│   ├── plant_knowledge_base.json    # Wikipedia data for 68 medicinal plants
│   ├── medicinal_classes.json       # Filtered list of 71 Nepal medicinal plants
│   └── plant_database_meta.json     # iNaturalist dataset metadata
│
├── .env.example                     # Environment variable template
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Team

| Member | Role |
|---|---|
| Triza Shashankar | RAG Pipeline + FastAPI Backend + Frontend Integration + Deployment |
| Sikha | ML Model (DINOv2 + FAISS) + Dataset + Plant Pipeline |



- [iNaturalist](https://www.inaturalist.org) — plant observation dataset
- [DINOv2](https://github.com/facebookresearch/dinov2) — Facebook AI vision model
- [Groq](https://groq.com) — LLM inference
- [ChromaDB](https://www.trychroma.com) — vector database
- [Wikipedia API](https://pypi.org/project/Wikipedia-API/) — plant knowledge source
