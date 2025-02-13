import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss
import numpy as np
from PIL import Image
import gradio as gr
import os

# Inicjalizacja modelu BLIP-2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


# Funkcja do ekstrakcji cech obrazu
def extract_image_features(image):
    """Ekstrahuje cechy obrazu jako wektor osadzeń."""
    # Konwersja obrazu do formatu RGB (jeśli nie jest w tym formacie)
    image = image.convert('RGB')

    # Przeskalowanie obrazu do rozmiaru 224x224 (wymagany rozmiar dla modelu BLIP)
    image = image.resize((224, 224))

    # Przetwarzanie obrazu za pomocą procesora BLIP
    inputs = processor(images=image, return_tensors="pt")

    # Ekstrakcja cech za pomocą modelu BLIP
    with torch.no_grad():
        features = model.vision_model(inputs.pixel_values)[1]  # Pobranie osadzeń wizualnych

    # Konwersja cech do formatu numpy i zwrócenie
    return features.numpy().astype('float32')


# Inicjalizacja zmiennych globalnych
index = None  # Indeks FAISS będzie inicjalizowany dynamicznie
image_embeddings = []
image_labels = []


# Funkcja dodawania wielu obrazów do bazy
def add_images_to_database(images):
    global index, image_embeddings, image_labels
    results = []

    for image in images:
        # Pobranie nazwy pliku (bez ścieżki)
        image_name = os.path.basename(image.name)

        # Otwarcie obrazu za pomocą PIL
        img = Image.open(image.name)

        # Ekstrakcja cech obrazu
        embed = extract_image_features(img)

        # Jeśli indeks FAISS nie został jeszcze utworzony, utwórz go z odpowiednim wymiarem
        if index is None:
            embedding_dim = embed.shape[1]  # Pobranie wymiaru osadzeń z pierwszego obrazu
            index = faiss.IndexFlatL2(embedding_dim)  # Inicjalizacja indeksu FAISS

        # Dodanie osadzeń i etykiety do bazy
        image_embeddings.append(embed)
        image_labels.append(image_name)
        index.add(embed)

        results.append(f"Dodano obraz: {image_name}")

    return "\n".join(results)


# Funkcja wyszukiwania w FAISS na podstawie nowego zdjęcia
def find_closest_image(query_image):
    query_embed = extract_image_features(query_image)
    D, I = index.search(query_embed, 1)  # Pobranie najbliższego sąsiada
    return image_labels[I[0][0]] if I[0][0] < len(image_labels) else "Nie znaleziono pasującego obrazu."


# Gradio UI
def image_search_ui(database_images, query_image):
    # Dodanie wielu obrazów do bazy
    add_images_to_database(database_images)

    # Wyszukanie najbliższego obrazu w bazie
    closest_image = find_closest_image(query_image)

    return f"Najbardziej podobna śrubka w bazie: {closest_image}"


# Tworzenie interfejsu Gradio
iface = gr.Interface(
    fn=image_search_ui,
    inputs=[
        gr.File(file_count="multiple", file_types=["image"]),  # Wiele obrazów jako pliki
        gr.Image(type="pil")  # Obraz do wyszukiwania
    ],
    outputs="text",
    title="System Rozpoznawania Śrubek",
    description="Dodaj wiele zdjęć śrubek do bazy i znajdź najbardziej podobną śrubkę w bazie."
)

# Uruchomienie interfejsu
if __name__ == "__main__":
    iface.launch()
