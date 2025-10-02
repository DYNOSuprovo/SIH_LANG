# Saksi Translation: Nepali-English Machine Translation

A machine translation project to translate text from Nepali to English using the NLLB (No Language Left Behind) model from Meta AI. The project includes scripts for data collection, text cleaning, model training, evaluation, and a REST API for serving the translation model.

## Table of Contents

- [Workflow](#workflow)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Project Structure](#project-structure)

## Workflow

The project follows a standard machine learning workflow for building a translation model:

1.  **Data Acquisition:** The process starts with collecting parallel text data. The `scripts/fetch_parallel_data.py` script is designed to download data from various online sources. This script can be extended to include more data sources to improve the model's performance.

2.  **Data Cleaning and Preprocessing:** Raw data from the web is often noisy. The `scripts/clean_text_data.py` script is used to perform several preprocessing steps, such as:
    *   Removing HTML tags and other artifacts.
    *   Normalizing Unicode characters.
    *   Filtering out sentences that are too long or too short.
    *   Ensuring a one-to-one correspondence between source and target sentences.

3.  **Model Finetuning:** The core of the project is finetuning a pre-trained NLLB (No Language Left Behind) model. The `src/train.py` script handles this process. It uses the Hugging Face `Trainer` class to manage the training loop, including:
    *   Loading the pre-trained model and tokenizer.
    *   Creating a PyTorch Dataset from the preprocessed data.
    *   Setting up training arguments, such as learning rate, batch size, and number of epochs.
    *   Running the training loop and saving the finetuned model.

4.  **Model Evaluation:** After training, it's crucial to evaluate the model's performance. The `src/evaluate.py` script calculates the BLEU (Bilingual Evaluation Understudy) score, a widely used metric for evaluating machine translation quality. This script compares the model's translations of a test set with reference translations.

5.  **Inference and Deployment:** Once the model is trained and evaluated, it can be used for translation. The `test_translation.py` script provides a simple example of how to load the finetuned model and translate a sentence. For a more practical application, the `api.py` script exposes the translation model as a REST API using FastAPI. This allows other applications to easily consume the translation service.

## Tech Stack

The technologies used in this project were chosen to create a robust and efficient machine translation pipeline:

-   **Python:** As the de facto language for machine learning, Python provides a rich ecosystem of libraries and frameworks.
-   **PyTorch:** Chosen for its flexibility and control over the model training process. Its dynamic computation graph is particularly useful for research and development.
-   **Hugging Face Transformers:** This library is the cornerstone of the project. It provides easy access to a vast number of pre-trained models, including NLLB, and a standardized interface for training and inference. This significantly reduces the amount of boilerplate code needed.
-   **Hugging Face Datasets:** This library simplifies the process of loading and preprocessing large datasets. It provides efficient data loading and manipulation capabilities, which are essential for training deep learning models.
-   **FastAPI:** For serving the model as an API, FastAPI was chosen for its high performance and ease of use. Its automatic generation of interactive API documentation (using Swagger UI) makes it easy to test and share the API.
-   **Uvicorn:** As a high-performance ASGI server, Uvicorn is the recommended server for running FastAPI applications in production.
-   **MLflow:** To ensure reproducibility and keep track of experiments, MLflow is used for logging training parameters, metrics, and model artifacts. This is crucial for managing the complexity of machine learning projects.

## Getting Started

### Prerequisites

-   **Python 3.10+:** Ensure you have a recent version of Python installed. You can download it from [python.org](https://www.python.org/).
-   **Git:** Git is required to clone the repository. You can download it from [git-scm.com](https://git-scm.com/).
-   **(Optional) NVIDIA GPU with CUDA:** For training the model, a GPU is highly recommended to speed up the process. Ensure you have the appropriate NVIDIA drivers and CUDA toolkit installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd saksi_translation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    ```
    -   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

-   **Fetch Parallel Data:**
    ```bash
    python scripts/fetch_parallel_data.py --output_dir data/raw
    ```
    This will download the parallel data to the `data/raw` directory.

-   **Clean Text Data:**
    ```bash
    python scripts/clean_text_data.py --input_dir data/raw --output_dir data/processed
    ```
    This will clean the raw data and save the processed files to the `data/processed` directory.

### Training

-   **Start Training:**
    ```bash
    python src/train.py \
        --model_name "facebook/nllb-200-distilled-600M" \
        --dataset_path "data/processed" \
        --output_dir "models/nllb-finetuned-nepali-en" \
        --learning_rate 2e-5 \
        --per_device_train_batch_size 8 \
        --num_train_epochs 3
    ```
    The training script will log experiments to MLflow. You can view the MLflow UI by running `mlflow ui` in a separate terminal.

### Evaluation

-   **Evaluate the Model:**
    ```bash
    python src/evaluate.py \
        --model_path "models/nllb-finetuned-nepali-en" \
        --test_data_path "data/test_sets/test.en" \
        --reference_data_path "data/test_sets/test.ne"
    ```
    This will print the BLEU score of the model on the test set.

### Translation

-   **Translate a Sentence:**
    Modify the `sample_text_to_translate` variable in `test_translation.py` with the sentence you want to translate. Then run:
    ```bash
    python test_translation.py
    ```

### API

-   **Run the API:**
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```
    The `--reload` flag can be added for development to automatically reload the server on code changes.

-   **API Documentation:**
    Once the API is running, you can access the interactive documentation at `http://127.0.0.1:8000/docs`.

-   **Example Request:**
    You can use `curl` or any other API client to send a POST request to the `/translate/` endpoint:
    ```bash
    curl -X POST "http://127.0.0.1:8000/translate/" \
         -H "Content-Type: application/json" \
         -d '{"text": "नेपाल एक सुन्दर देश हो।"}'
    ```
    **Expected Response:**
    ```json
    {
      "translated_text": "Nepal is a beautiful country."
    }
    ```

## File Descriptions

-   `api.py`: Defines the FastAPI application that serves the translation model.
-   `test_translation.py`: A simple script for testing the translation of a single sentence.
-   `src/train.py`: The main script for finetuning the NLLB model.
-   `src/evaluate.py`: A script for evaluating the performance of the finetuned model.
-   `src/translate.py`: Contains the core translation logic.
-   `scripts/fetch_parallel_data.py`: A script for downloading parallel data from the web.
-   `scripts/clean_text_data.py`: A script for cleaning and preprocessing the raw text data.
-   `requirements.txt`: A list of all the Python dependencies required for the project.
-   `mlruns/`: The directory where MLflow stores experiment tracking data.
-   `models/`: The directory where the finetuned models are saved.

## Project Structure

```
saksi_translation/
├── .gitignore
├── api.py                  # FastAPI application for serving the model
├── api_log.txt             # Log file for the API
├── baseline_translate.py   # Baseline translation script
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── test_translation.py     # Script for testing the translation model
├── data/
│   ├── processed/          # Processed data for training
│   ├── raw/                # Raw data downloaded from the web
│   └── test_sets/          # Test sets for evaluation
├── mlruns/                 # MLflow experiment tracking data
├── models/
│   └── nllb-finetuned-nepali-en/ # Finetuned model
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/
│   ├── clean_text_data.py
│   ├── create_test_set.py
│   ├── download_model.py
│   ├── fetch_parallel_data.py
│   └── scrape_bbc_nepali.py
└── src/
    ├── __init__.py
    ├── evaluate.py         # Script for evaluating the model
    ├── train.py            # Script for training the model
    └── translate.py        # Script for translating text
```