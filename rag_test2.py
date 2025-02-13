import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss
import numpy as np
from PIL import Image
import gradio as gr

# Inicjalizacja modelu BLIP-2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Funkcja do ekstrakcji cech obrazu
def extract_image_features(image):
    """Ekstrahuje cechy obrazu jako wektor osadzeń."""
    image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.vision_model(inputs.pixel_values)[1]  # Pobranie osadzeń wizualnych
    return features.numpy().astype('float32')

# Sprawdzenie wymiaru osadzeń
test_image = Image.new('RGB', (224, 224))  # Przykładowy obraz do testu
test_embed = extract_image_features(test_image)
embedding_dim = test_embed.shape[1]  # Pobranie wymiaru osadzeń

# Tworzenie FAISS dla bazy obrazów
index = faiss.IndexFlatL2(embedding_dim)  # Dostosowanie wymiaru indeksu FAISS
image_embeddings = []
image_labels = []

# Funkcja dodawania obrazu do bazy
def add_image_to_database(image, image_name):
    global index, image_embeddings, image_labels
    embed = extract_image_features(image)
    if embed.shape[1] != embedding_dim:
        raise ValueError(f"Wymiar osadzeń ({embed.shape[1]}) nie zgadza się z wymiarami indeksu FAISS ({embedding_dim}).")
    image_embeddings.append(embed)
    image_labels.append(image_name)
    index.add(embed)
    return f"Dodano obraz: {image_name}"

# Funkcja wyszukiwania w FAISS na podstawie nowego zdjęcia
def find_closest_image(query_image):
    query_embed = extract_image_features(query_image)
    D, I = index.search(query_embed, 1)  # Pobranie najbliższego sąsiada
    return image_labels[I[0][0]] if I[0][0] < len(image_labels) else "Nie znaleziono pasującego obrazu."

# Gradio UI
def image_search_ui(database_image, query_image):
    add_image_to_database(database_image, "nowy_obiekt")
    closest_image = find_closest_image(query_image)
    return f"Najbardziej podobna śrubka w bazie: {closest_image}"

iface = gr.Interface(
    fn=image_search_ui,
    inputs=[gr.Image(type="pil"), gr.Image(type="pil")],
    outputs="text",
    title="System Rozpoznawania Śrubek",
    description="Dodaj zdjęcie śrubki do bazy i znajdź najbardziej podobną do niej śrubkę w bazie."
)

if __name__ == "__main__":
    iface.launch()