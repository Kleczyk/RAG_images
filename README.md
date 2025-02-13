# Screw Recognition System

This project is a simple image-based screw recognition system that uses the BLIP model for feature extraction and FAISS for similarity search. It allows you to add multiple images of screws to a database and find the most similar screw in the database based on a query image.

## Requirements

To run this project, you need the following:

- Python 3.8 or higher

## Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:Kleczyk/screw-recognition-system.git
   cd screw-recognition-system
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

   Alternatively, you can install the dependencies manually:
   ```bash
   pip install torch transformers faiss-cpu gradio pillow numpy
   ```

## Usage

1. **Run the application**:
   ```bash
   python rag_test2.py
   ```

2. **Access the Gradio interface**:
   - After running the script, a Gradio interface will be launched.
   - Open your web browser and navigate to the URL provided in the terminal (usually `http://127.0.0.1:7860`).

3. **Add images to the database**:
   - Use the "Upload Images" section to upload multiple images of screws. The images will be added to the database, and their filenames will be used as labels.

4. **Search for similar screws**:
   - Upload a query image in the "Query Image" section.
   - The system will return the name of the most similar screw in the database.

## Example

1. Upload multiple images of screws (e.g., `screw1.jpg`, `screw2.jpg`, `screw3.jpg`).
2. Upload a query image (e.g., `query_screw.jpg`).
3. The system will display the name of the most similar screw in the database (e.g., `screw2.jpg`).

## Code Structure

- `rag_test2.py`: The main script that contains the Gradio interface and the logic for feature extraction and similarity search.
- `README.md`: This file, containing instructions on how to set up and use the project.

## Dependencies

- `torch`: For deep learning operations.
- `transformers`: For using the BLIP model.
- `faiss-cpu`: For efficient similarity search.
- `gradio`: For creating the web interface.
- `pillow`: For image processing.
- `numpy`: For numerical operations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
