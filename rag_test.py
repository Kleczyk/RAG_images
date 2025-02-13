import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss
import numpy as np
from openai import OpenAI
from PIL import Image

# Inicjalizacja modelu BLIP-2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(image_path):
    """Generuje opis obrazu przy użyciu BLIP-2."""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Przykładowe śrubki do bazy danych (wektorowej)
knowledge_base = {
    "śrubka M3x10 krzyżowa": "Mała śrubka z gwintem M3, długość 10mm, łeb krzyżowy PH2.",
    "śrubka M4x12 torx": "Śrubka z gwintem M4, długość 12mm, łeb torx T20.",
    "śrubka samogwintująca 3.5x9.5": "Mała śrubka samogwintująca, długość 9.5mm, szerokość 3.5mm, łeb płaski."
}

# Tworzenie FAISS dla bazy śrubek
index = faiss.IndexFlatL2(768)  # 768 to wymiar osadzeń BLIP-2
screw_embeddings = []
screw_labels = []

# Konwersja opisów na osadzenia
for label, description in knowledge_base.items():
    inputs = processor(description, return_tensors="pt")
    with torch.no_grad():
        embed = model.get_input_embeddings()(inputs.input_ids).mean(dim=1).numpy()
    screw_embeddings.append(embed)
    screw_labels.append(label)

# Dodanie do FAISS
screw_embeddings = np.vstack(screw_embeddings).astype('float32')
index.add(screw_embeddings)

# Funkcja wyszukiwania w FAISS
def find_closest_screw(description):
    """Znajduje najbardziej podobną śrubkę w bazie."""
    inputs = processor(description, return_tensors="pt")
    with torch.no_grad():
        embed = model.get_input_embeddings()(inputs.input_ids).mean(dim=1).numpy().astype('float32')
    D, I = index.search(embed, 1)  # Pobranie najbliższego sąsiada
    return screw_labels[I[0][0]] if I[0][0] < len(screw_labels) else "Nie znaleziono pasującej śrubki."

# OpenAI GPT-4 do generowania odpowiedzi
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_response(screw_type, description):
    """Generuje odpowiedź końcową."""
    prompt = f"Obiekt na zdjęciu wygląda na {screw_type}. Opis: {description}. Powiedz więcej o tej śrubce."
    response = client.Completion.create(
        engine="gpt-4-turbo",
        prompt=prompt,
        max_tokens=100
    )
    return response['choices'][0]['text']

# DEMO: Testowanie na nowym obrazie
image_path = "test_screw.jpg"  # Podmień na swoje zdjęcie śrubki
caption = generate_caption(image_path)
print("Opis obrazu:", caption)

# Znalezienie podobnej śrubki
closest_screw = find_closest_screw(caption)
print("Najbliższa śrubka w bazie:", closest_screw)

# Generowanie odpowiedzi przez GPT-4
final_response = generate_response(closest_screw, caption)
print("Odpowiedź GPT-4:", final_response)
