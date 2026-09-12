import json

INPUT_FILE = "data/asl/WLASL_v0.3.json"
OUTPUT_FILE = "data/asl/WLASL100.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# WLASL100 = les 100 premiers glosses
wlasl100 = data[:100]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(wlasl100, f, indent=4)

print("WLASL100 créé")
print("Nombre de classes :", len(wlasl100))

print("\nPremières classes :")

for item in wlasl100[:10]:
    print(item["gloss"])
