import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss
import numpy as np
from PIL import Image
import gradio as gr
import os

# Initialization of the BLIP-2 model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


# Function to extract image features
def extract_image_features(image):
    """Extracts image features as an embedding vector."""
    # Convert image to RGB format (if not already in this format)
    image = image.convert('RGB')

    # Resize the image to 224x224 (required size for the BLIP model)
    image = image.resize((224, 224))

    # Process the image using the BLIP processor
    inputs = processor(images=image, return_tensors="pt")

    # Extract features using the BLIP model
    with torch.no_grad():
        features = model.vision_model(inputs.pixel_values)[1]  # Retrieve visual embeddings

    # Convert features to numpy format and return
    return features.numpy().astype('float32')


# Initialization of global variables
index = None  # FAISS index will be initialized dynamically
image_embeddings = []
image_labels = []


# Function to add multiple images to the database
def add_images_to_database(images):
    global index, image_embeddings, image_labels
    results = []

    for image in images:
        # Get the file name (without path)
        image_name = os.path.basename(image.name)

        # Open the image using PIL
        img = Image.open(image.name)

        # Extract image features
        embed = extract_image_features(img)

        # If FAISS index has not been created yet, initialize it with the correct dimension
        if index is None:
            embedding_dim = embed.shape[1]  # Get embedding dimension from the first image
            index = faiss.IndexFlatL2(embedding_dim)  # Initialize FAISS index

        # Add embeddings and labels to the database
        image_embeddings.append(embed)
        image_labels.append(image_name)
        index.add(embed)

        results.append(f"Added image: {image_name}")

    return "\n".join(results)


# Function to search FAISS based on a new image
def find_closest_image(query_image):
    query_embed = extract_image_features(query_image)
    D, I = index.search(query_embed, 1)  # Retrieve the closest neighbor
    return image_labels[I[0][0]] if I[0][0] < len(image_labels) else "No matching image found."


# Gradio UI
def image_search_ui(database_images, query_image):
    # Add multiple images to the database
    add_images_to_database(database_images)

    # Find the closest image in the database
    closest_image = find_closest_image(query_image)

    return f"Most similar screw in the database: {closest_image}"


# Creating the Gradio interface
iface = gr.Interface(
    fn=image_search_ui,
    inputs=[
        gr.File(file_count="multiple", file_types=["image"]),  # Multiple images as files
        gr.Image(type="pil")  # Image for searching
    ],
    outputs="text",
    title="Screw Recognition System",
    description="Add multiple images of screws to the database and find the most similar screw in the database."
)

# Launching the interface
if __name__ == "__main__":
    iface.launch(share=True)
