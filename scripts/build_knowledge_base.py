import json
import time
import re

try:
    import wikipediaapi
except ModuleNotFoundError:
    print("Install wikipedia-api: pip install wikipedia-api")
    exit(1)

# Common names mapping for better Wikipedia search
COMMON_NAMES = {
    "Ocimum_tenuiflorum": "Tulsi",
    "Azadirachta_indica": "Neem",
    "Zingiber_officinale": "Ginger",
    "Curcuma_longa": "Turmeric",
    "Aloe_vera": "Aloe vera",
    "Aconitum_heterophyllum": "Atis root",
    "Nardostachys_jatamansi": "Spikenard",
    "Swertia_chirayita": "Chirayito",
    "Berberis_asiatica": "Berberis asiatica",
    "Rhododendron_arboreum": "Rhododendron arboreum",
    "Terminalia_chebula": "Haritaki",
    "Terminalia_bellirica": "Bibhitaki",
    "Phyllanthus_emblica": "Amla",
    "Withania_somnifera": "Ashwagandha",
    "Tinospora_cordifolia": "Guduchi",
    "Asparagus_racemosus": "Shatavari",
    "Piper_longum": "Long pepper",
    "Piper_nigrum": "Black pepper",
    "Elettaria_cardamomum": "Cardamom",
    "Cinnamomum_tamala": "Indian bay leaf",
    "Valeriana_jatamansi": "Valerian jatamansi",
    "Cannabis_sativa": "Cannabis sativa",
    "Podophyllum_hexandrum": "Himalayan mayapple",
    "Dactylorhiza_hatagirea": "Dactylorhiza hatagirea",
    "Juglans_regia": "Walnut",
    "Ficus_religiosa": "Sacred fig",
    "Ficus_benghalensis": "Banyan tree",
    "Moringa_oleifera": "Moringa",
    "Artemisia_indica": "Mugwort",
    "Calotropis_gigantea": "Crown flower",
    "Centella_asiatica": "Gotu kola",
    "Cuscuta_reflexa": "Dodder",
    "Datura_stramonium": "Jimsonweed",
    "Mentha_arvensis": "Corn mint",
    "Mimosa_pudica": "Sensitive plant",
    "Plantago_major": "Broadleaf plantain",
    "Punica_granatum": "Pomegranate",
    "Ricinus_communis": "Castor oil plant",
    "Rubia_cordifolia": "Indian madder",
    "Urtica_dioica": "Stinging nettle",
    "Zanthoxylum_armatum": "Zanthoxylum armatum",
    "Bergenia_ciliata": "Bergenia ciliata",
    "Ephedra_gerardiana": "Ephedra gerardiana",
    "Hippophae_rhamnoides": "Sea buckthorn",
    "Meconopsis_napaulensis": "Nepal poppy",
    "Picrorhiza_kurrooa": "Kutki",
    "Rheum_australe": "Himalayan rhubarb",
    "Taxus_wallichiana": "Himalayan yew",
    "Acorus_calamus": "Sweet flag",
    "Aegle_marmelos": "Bael",
    "Allium_sativum": "Garlic",
    "Allium_cepa": "Onion",
    "Andrographis_paniculata": "Andrographis",
    "Bacopa_monnieri": "Brahmi",
    "Camellia_sinensis": "Tea plant",
    "Capsicum_annuum": "Capsicum",
    "Coriandrum_sativum": "Coriander",
    "Cuminum_cyminum": "Cumin",
    "Foeniculum_vulgare": "Fennel",
    "Leucas_aspera": "Leucas aspera",
    "Ocimum_basilicum": "Basil",
    "Origanum_vulgare": "Oregano",
    "Oxalis_corniculata": "Creeping woodsorrel",
    "Pinus_roxburghii": "Chir pine",
    "Rhododendron_lepidotum": "Rhododendron lepidotum",
    "Rumex_nepalensis": "Nepal dock",
    "Sapindus_mukorossi": "Soapnut",
    "Solanum_nigrum": "Black nightshade",
    "Thalictrum_foliolosum": "Thalictrum foliolosum",
    "Vitex_negundo": "Five-leaved chaste tree",
    "Woodfordia_fruticosa": "Woodfordia fruticosa",
}

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text[:3000]  # limit chunk size

def fetch_plant_info(latin_name, common_name, wiki):
    search_name = latin_name.replace('_', ' ')
    page = wiki.page(search_name)

    if not page.exists():
        # try common name
        page = wiki.page(common_name)

    if not page.exists():
        return None

    summary = clean_text(page.summary)
    if len(summary) < 50:
        return None

    return summary

def build_text_chunk(latin_name, common_name, wiki_text):
    display_name = latin_name.replace('_', ' ')
    chunk = f"""Plant: {display_name}
Common Name: {common_name}
Scientific Name: {display_name}

{wiki_text}"""
    return chunk

def main():
    # Load matched plants
    with open('data/medicinal_classes.json') as f:
        matched_plants = json.load(f)

    print(f"Fetching Wikipedia info for {len(matched_plants)} plants...")
    print("This may take 2-3 minutes...\n")

    wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='MedicinalPlantAI/1.0 (Herald College Kathmandu)'
    )

    knowledge_base = []
    failed = []

    for i, latin_name in enumerate(matched_plants):
        common_name = COMMON_NAMES.get(latin_name, latin_name.replace('_', ' '))
        print(f"[{i+1}/{len(matched_plants)}] Fetching: {latin_name} ({common_name})")

        wiki_text = fetch_plant_info(latin_name, common_name, wiki)

        if wiki_text:
            chunk = build_text_chunk(latin_name, common_name, wiki_text)
            knowledge_base.append({
                "latin_name": latin_name,
                "common_name": common_name,
                "text": chunk
            })
            print(f"  ✅ Got {len(wiki_text)} chars")
        else:
            failed.append(latin_name)
            print(f"  ❌ Not found on Wikipedia")

        time.sleep(0.5)  # be polite to Wikipedia API

    # Save knowledge base
    with open('plant_knowledge_base.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"Successfully fetched: {len(knowledge_base)} plants")
    print(f"Failed: {len(failed)} plants")
    if failed:
        print(f"Failed plants: {failed}")
    print(f"Knowledge base saved to: plant_knowledge_base.json")

if __name__ == "__main__":
    main()
