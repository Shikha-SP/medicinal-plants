import json

# Known Nepal medicinal plants in Latin scientific names
NEPAL_MEDICINAL_PLANTS = [
    "Ocimum_tenuiflorum",       # Tulsi
    "Azadirachta_indica",       # Neem
    "Zingiber_officinale",      # Ginger
    "Curcuma_longa",            # Turmeric
    "Aloe_vera",                # Aloe Vera
    "Aconitum_heterophyllum",   # Atis root
    "Nardostachys_jatamansi",   # Spikenard
    "Swertia_chirayita",        # Chirayito
    "Berberis_asiatica",        # Chutro
    "Rhododendron_arboreum",    # Lali Gurans
    "Terminalia_chebula",       # Harro
    "Terminalia_bellirica",     # Barro
    "Phyllanthus_emblica",      # Amala
    "Withania_somnifera",       # Ashwagandha
    "Tinospora_cordifolia",     # Gurjo
    "Asparagus_racemosus",      # Satawari
    "Glycyrrhiza_glabra",       # Jethimadhu
    "Piper_longum",             # Pipla
    "Piper_nigrum",             # Black Pepper
    "Elettaria_cardamomum",     # Cardamom
    "Cinnamomum_tamala",        # Tejpat
    "Valeriana_jatamansi",      # Sugandhawal
    "Atropa_belladonna",        # Belladonna
    "Cannabis_sativa",          # Hemp
    "Podophyllum_hexandrum",    # Bankakri
    "Dactylorhiza_hatagirea",   # Panchaunle
    "Juglans_regia",            # Okhar
    "Ficus_religiosa",          # Pipal
    "Ficus_benghalensis",       # Barh
    "Moringa_oleifera",         # Sajiwan
    "Adhatoda_vasica",          # Asuro
    "Artemisia_indica",         # Titepati
    "Calotropis_gigantea",      # Aank
    "Cassia_tora",              # Chakramard
    "Centella_asiatica",        # Gotu Kola
    "Cuscuta_reflexa",          # Akashe lahara
    "Datura_stramonium",        # Dhaturo
    "Mentha_arvensis",          # Pudina
    "Mimosa_pudica",            # Lajawanti
    "Plantago_major",           # Isabgol
    "Punica_granatum",          # Pomegranate
    "Ricinus_communis",         # Aandi
    "Rubia_cordifolia",         # Majitho
    "Senna_alexandrina",        # Sonamukhi
    "Urtica_dioica",            # Sisnu
    "Viscum_album",             # Ainjeru
    "Zanthoxylum_armatum",      # Timur
    "Aconitum_spicatum",        # Bikh
    "Bergenia_ciliata",         # Pakhanbed
    "Coptis_teeta",             # Mamira
    "Ephedra_gerardiana",       # Somlata
    "Gentiana_kurroo",          # Kutki
    "Hippophae_rhamnoides",     # Sea Buckthorn
    "Inula_racemosa",           # Pushkarmool
    "Meconopsis_napaulensis",   # Seto Khukuri Phool
    "Picrorhiza_kurrooa",       # Kutki
    "Rheum_australe",           # Padamchal
    "Saussurea_costus",         # Kuth
    "Taxus_wallichiana",        # Lauth Salla
    "Acorus_calamus",           # Bojho
    "Aegle_marmelos",           # Bel
    "Allium_sativum",           # Garlic
    "Allium_cepa",              # Onion
    "Andrographis_paniculata",  # Kalmegh
    "Bacopa_monnieri",          # Brahmi
    "Calamus_rotang",           # Bet
    "Camellia_sinensis",        # Tea
    "Capsicum_annuum",          # Khursani
    "Coriandrum_sativum",       # Dhaniya
    "Cuminum_cyminum",          # Jeera
    "Emblica_officinalis",      # Amla
    "Foeniculum_vulgare",       # Saunf
    "Lawsonia_inermis",         # Mehendi
    "Leucas_aspera",            # Dronapushpi
    "Nymphaea_stellata",        # Blue Lotus
    "Ocimum_basilicum",         # Babari
    "Origanum_vulgare",         # Oregano
    "Oxalis_corniculata",       # Chariamilo
    "Phyllanthus_niruri",       # Bhuiamla
    "Pinus_roxburghii",         # Salla
    "Rhododendron_lepidotum",   # Bhale Sunpati
    "Rumex_nepalensis",         # Halhale
    "Sapindus_mukorossi",       # Ritha
    "Solanum_nigrum",           # Kakaмaro
    "Thalictrum_foliolosum",    # Pirre
    "Vitex_negundo",            # Simali
    "Woodfordia_fruticosa",     # Dhairo
]

# Load Sikha's dataset classes
with open('data/plant_database_meta.json', encoding='utf-8') as f:
    data = json.load(f)

sikha_classes = set(item['class'] for item in data)

# Find matches
matched = []
not_found = []

for plant in NEPAL_MEDICINAL_PLANTS:
    if plant in sikha_classes:
        matched.append(plant)
    else:
        not_found.append(plant)

print(f"Total medicinal plants checked: {len(NEPAL_MEDICINAL_PLANTS)}")
print(f"Found in Sikha's dataset: {len(matched)}")
print(f"Not found: {len(not_found)}")
print()
print("MATCHED PLANTS (use these for ChromaDB):")
for p in matched:
    print(f"  ✅ {p}")
print()
print("NOT FOUND:")
for p in not_found:
    print(f"  ❌ {p}")

# Save matched list
with open('medicinal_classes.json', 'w') as f:
    json.dump(matched, f, indent=2)
print()
print(f"Saved matched list to medicinal_classes.json")
