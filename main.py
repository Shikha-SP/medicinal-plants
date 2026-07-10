import os
import json
import re
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from predict import predict_plant

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    from dotenv import load_dotenv
except ModuleNotFoundError as e:
    print(f"Missing package: {e.name}")
    print("Install with: pip install fastapi uvicorn groq chromadb sentence-transformers python-multipart python-dotenv")
    exit(1)

# ── Load environment variables ───────────────────────────
load_dotenv()

# ── Config ──────────────────────────────────────────────
CHROMA_DIR = "data/chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
CONFIDENCE_THRESHOLD = 0.4
TOP_K_RETRIEVE = 5

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

# ── Common names lookup ───────────────────────────────────
COMMON_NAMES = {
    "Ocimum_tenuiflorum": "Tulsi",
    "Azadirachta_indica": "Neem",
    "Zingiber_officinale": "Ginger",
    "Curcuma_longa": "Turmeric",
    "Aloe_vera": "Aloe Vera",
    "Aconitum_heterophyllum": "Atis Root",
    "Nardostachys_jatamansi": "Spikenard",
    "Swertia_chirayita": "Chirayito",
    "Berberis_asiatica": "Chutro",
    "Rhododendron_arboreum": "Lali Gurans",
    "Terminalia_chebula": "Harro",
    "Terminalia_bellirica": "Barro",
    "Phyllanthus_emblica": "Amala",
    "Withania_somnifera": "Ashwagandha",
    "Tinospora_cordifolia": "Gurjo",
    "Asparagus_racemosus": "Satawari",
    "Piper_longum": "Pipla",
    "Piper_nigrum": "Black Pepper",
    "Elettaria_cardamomum": "Cardamom",
    "Cinnamomum_tamala": "Tejpat",
    "Valeriana_jatamansi": "Sugandhawal",
    "Cannabis_sativa": "Hemp",
    "Podophyllum_hexandrum": "Bankakri",
    "Dactylorhiza_hatagirea": "Panchaunle",
    "Juglans_regia": "Okhar",
    "Ficus_religiosa": "Pipal",
    "Ficus_benghalensis": "Barh",
    "Moringa_oleifera": "Sajiwan",
    "Artemisia_indica": "Titepati",
    "Calotropis_gigantea": "Aank",
    "Centella_asiatica": "Gotu Kola",
    "Cuscuta_reflexa": "Akashe Lahara",
    "Datura_stramonium": "Dhaturo",
    "Mentha_arvensis": "Pudina",
    "Mimosa_pudica": "Lajawanti",
    "Plantago_major": "Isabgol",
    "Punica_granatum": "Anar",
    "Ricinus_communis": "Aandi",
    "Rubia_cordifolia": "Majitho",
    "Urtica_dioica": "Sisnu",
    "Zanthoxylum_armatum": "Timur",
    "Bergenia_ciliata": "Pakhanbed",
    "Ephedra_gerardiana": "Somlata",
    "Hippophae_rhamnoides": "Sea Buckthorn",
    "Meconopsis_napaulensis": "Nepal Poppy",
    "Picrorhiza_kurrooa": "Kutki",
    "Rheum_australe": "Padamchal",
    "Taxus_wallichiana": "Lauth Salla",
    "Acorus_calamus": "Bojho",
    "Aegle_marmelos": "Bel",
    "Allium_sativum": "Garlic",
    "Allium_cepa": "Pyaj",
    "Andrographis_paniculata": "Kalmegh",
    "Bacopa_monnieri": "Brahmi",
    "Camellia_sinensis": "Chiya",
    "Capsicum_annuum": "Khursani",
    "Coriandrum_sativum": "Dhaniya",
    "Cuminum_cyminum": "Jeera",
    "Foeniculum_vulgare": "Saunf",
    "Leucas_aspera": "Dronapushpi",
    "Ocimum_basilicum": "Babari",
    "Origanum_vulgare": "Oregano",
    "Oxalis_corniculata": "Chariamilo",
    "Pinus_roxburghii": "Salla",
    "Sapindus_mukorossi": "Ritha",
    "Solanum_nigrum": "Kakamaro",
    "Vitex_negundo": "Simali",
    "Woodfordia_fruticosa": "Dhairo",
}

# ── Load models and DB at startup ───────────────────────
print("[startup] Loading sentence-transformers model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print("[startup] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("medicinal_plants")
print(f"[startup] ChromaDB ready — {collection.count()} plants loaded")

print("[startup] Connecting to Groq...")
groq_client = Groq(api_key=GROQ_API_KEY)
print("[startup] All systems ready")

# ── FastAPI app ──────────────────────────────────────────
app = FastAPI(
    title="Nepal Medicinal Plant Intelligence System",
    description="AI-powered plant identification and medicinal knowledge API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

# ── Mock ML prediction ────────────────────────────────────
def predict_plant_mock(image: Image.Image) -> dict:
    """
    MOCK FUNCTION — replace with Sikha's real FAISS search
    when plant_database.index is ready.

    Replace this entire function body with:
    from plant_pipeline.query_index import search_plant
    result = search_plant(image)
    return result
    """
    return {
        "plant_name": "Ocimum_tenuiflorum",
        "confidence": 0.94
    }

# ── Improved RAG query ────────────────────────────────────
def query_knowledge_base(plant_name: str, common_name: str) -> str:
    """
    Query ChromaDB with richer query for better retrieval.
    Retrieves top 5, filters by relevance, returns best 2.
    """
    display_name = plant_name.replace('_', ' ')
    query = (
        f"{display_name} {common_name} medicinal uses "
        f"traditional medicine Nepal Ayurveda safety properties"
    )

    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_RETRIEVE, collection.count())
    )

    if not results['documents'][0]:
        return None

    docs = results['documents'][0]
    plant_key = display_name.lower()
    common_key = common_name.lower()

    # Keep only chunks that mention the plant
    relevant = [
        doc for doc in docs
        if plant_key in doc.lower() or common_key in doc.lower()
    ]

    if not relevant:
        return None

    best_docs = relevant[:2]
    return "\n\n".join(best_docs)

# ── Groq: grounded generation ─────────────────────────────
def generate_plant_report(plant_name: str, common_name: str, context: str) -> dict:
    """Generate report grounded in ChromaDB retrieved context."""
    display_name = plant_name.replace('_', ' ')

    prompt = f"""You are a Nepal medicinal plant expert. Use the information below to answer about {display_name} (common name: {common_name}).

Source information:
{context}

Return a JSON object with exactly these fields and real values:
{{
    "plant_name": "{common_name}",
    "nepali_name": "Nepali local name if known, else empty string",
    "latin_name": "{display_name}",
    "medicinal_uses": "Describe the specific medicinal uses of this plant in 2-3 sentences based on the source",
    "safety": "Must be exactly one of: SAFE, USE WITH CAUTION, or TOXIC",
    "safety_note": "Explain the safety classification briefly",
    "location_in_nepal": "Specific regions of Nepal where this plant is found such as Terai, Mid-hills, High Himalaya, or specific districts",
    "traditional_use": "How this plant is traditionally used in Nepali or Ayurvedic medicine",
    "source": "Wikipedia"
}}

Important: Return only valid JSON. All field values must be real information, not placeholder text."""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700
    )

    raw = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        cleaned = match.group()
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', cleaned)
        return json.loads(cleaned)
    raise ValueError("Groq did not return valid JSON")

# ── Groq: fallback generation ─────────────────────────────
def generate_report_fallback(plant_name: str, common_name: str) -> dict:
    """Fallback when plant is not in ChromaDB. Uses Groq general knowledge."""
    display_name = plant_name.replace('_', ' ')

    prompt = f"""You are a Nepal medicinal plant expert. Generate accurate information about {display_name} (common name: {common_name}).

Return JSON with exactly these fields:
{{
    "plant_name": "{common_name}",
    "nepali_name": "Nepali name if known, else empty string",
    "latin_name": "{display_name}",
    "medicinal_uses": "main medicinal uses, 2-3 sentences",
    "safety": "SAFE or USE WITH CAUTION or TOXIC",
    "safety_note": "brief safety explanation",
    "location_in_nepal": "where this plant grows in Nepal — mention specific regions like Terai, Mid-hills, High Himalaya, or specific districts if known. If exact Nepal location unknown, state the general altitude or climate zone where it grows",
    "traditional_use": "use in traditional Nepali medicine",
    "source": "state the source of this information e.g. Ayurvedic texts, WHO monographs, traditional Nepali medicine knowledge, etc."
}}

Important: Return only valid JSON. All field values must be real information, not placeholder text."""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )

    raw = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        cleaned = match.group()
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', cleaned)
        return json.loads(cleaned)
    raise ValueError("Groq did not return valid JSON")

# ── Routes ───────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "plants_in_db": collection.count(),
        "model_status": "active — Sikha's FAISS model integrated",
        "rag_status": "active",
        "groq_status": "active"
    }

@app.post("/identify")
async def identify_plant(file: UploadFile = File(...)):
    """
    Main endpoint — receives plant image, returns medicinal report.

    Flow:
    1. Validate and read image
    2. ML model prediction
    3. Confidence threshold check
    4. Query ChromaDB
    5. Generate report with Groq
    6. Return JSON
    """

    # Step 1 — Validate image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    # Step 2 — ML prediction
    try:
        prediction = predict_plant(image)
        plant_name = prediction["plant_name"]
        confidence = prediction["confidence"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plant identification failed: {str(e)}")

    # Step 3 — Confidence check
    if confidence < CONFIDENCE_THRESHOLD:
        return JSONResponse(content={
            "success": False,
            "confidence": confidence,
            "message": "Could not identify plant clearly. Please try a clearer photo.",
            "report": None
        })

    # Step 4 — Get common name
    common_name = COMMON_NAMES.get(plant_name, plant_name.replace('_', ' '))

    # Step 5 — Query ChromaDB
    context = query_knowledge_base(plant_name, common_name)

    # Step 6 — Generate report
    try:
        if context:
            report = generate_plant_report(plant_name, common_name, context)
            report_source = "knowledge_base"
        else:
            report = generate_report_fallback(plant_name, common_name)
            report_source = "general_knowledge"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    # Step 7 — Return response
    return JSONResponse(content={
        "success": True,
        "confidence": round(confidence, 3),
        "ml_prediction": plant_name,
        "report_source": report_source,
        "report": report
    })

# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)