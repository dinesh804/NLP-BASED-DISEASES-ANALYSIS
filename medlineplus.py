import requests
from pathlib import Path

URL = "https://wsearch.nlm.nih.gov/ws/query"

params = {
    "db": "healthTopics",
    "term": "dengue",
    "rettype": "topic",
    "retmax": 20,
}

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "medlineplus_dengue.xml"

print("Downloading MedlinePlus data...")

response = requests.get(
    URL,
    params=params,
    timeout=60
)

response.raise_for_status()

output_file.write_bytes(response.content)

print("Download successful!")
print(f"Saved to: {output_file}")
print(f"Size: {output_file.stat().st_size / 1024:.2f} KB")