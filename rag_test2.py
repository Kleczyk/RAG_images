import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss
import numpy as np
from PIL import Image

# Inicjalizacja modelu BLIP-2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def extract_image_features(image_path):
    """Ekstrahuje cechy obrazu jako wektor osadzeń."""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embed = model.get_input_embeddings()(inputs.input_ids).mean(dim=1).numpy().astype('float32')
    return embed

# Przykładowe obrazy śrubek do bazy danych
image_paths = ["screw_m3.jpg", "screw_m4.jpg", "screw_self_tapping.jpg"]  # Podmień na rzeczywiste ścieżki

# Tworzenie FAISS dla bazy obrazów
index = faiss.IndexFlatL2(768)  # 768 to wymiar osadzeń BLIP-2
image_embeddings = []
image_labels = []

# Przetwarzanie obrazów i dodawanie ich wektorów do FAISS
for img_path in image_paths:
    embed = extract_image_features(img_path)
    image_embeddings.append(embed)
    image_labels.append(img_path)

# Dodanie do FAISS
image_embeddings = np.vstack(image_embeddings).astype('float32')
index.add(image_embeddings)

# Funkcja wyszukiwania w FAISS na podstawie nowego zdjęcia
def find_closest_image(query_image_path):
    """Znajduje najbardziej podobny obraz w bazie."""
    query_embed = extract_image_features(query_image_path)
    D, I = index.search(query_embed, 1)  # Pobranie najbliższego sąsiada
    return image_labels[I[0][0]] if I[0][0] < len(image_labels) else "Nie znaleziono pasującego obrazu."

# DEMO: Testowanie na nowym obrazie
query_image_path = "test_screw.jpg"  # Podmień na swoje zdjęcie śrubki
closest_image = find_closest_image(query_image_path)
print("Najbardziej podobna śrubka w bazie:", closest_image)
