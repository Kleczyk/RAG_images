import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss
import numpy as np
from PIL import Image
import gradio as gr

# Inicjalizacja modelu BLIP-2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def extract_image_features(image):
    """Ekstrahuje cechy obrazu jako wektor osadzeń."""
    image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embed = model.get_input_embeddings()(inputs.input_ids).mean(dim=1).numpy().astype('float32')
    return embed

# Tworzenie FAISS dla bazy obrazów
index = faiss.IndexFlatL2(768)  # 768 to wymiar osadzeń BLIP-2
image_embeddings = []
image_labels = []

# Funkcja dodawania obrazu do bazy

def add_image_to_database(image, image_name):
    global index, image_embeddings, image_labels
    embed = extract_image_features(image)
    image_embeddings.append(embed)
    image_labels.append(image_name)
    index.add(np.array(embed, dtype='float32').reshape(1, -1))
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
