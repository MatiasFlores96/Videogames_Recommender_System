# Hybrid Videogame Recommender System

This repository contains the source code for the Master's thesis **"Hybrid Videogame Recommender System Integrating User Behavior, Game Attributes, and Community Ratings"** by Matías Flores (ITBA).

The main goal is to develop and compare videogame recommender systems for the Steam platform, evaluating a traditional **collaborative filtering** model against a **hybrid** model that integrates collaborative filtering, content-based, and sentiment analysis approaches, all implemented with KerasRS.

## 🚀 Project Goals

## 🎯 Thesis Objectives

- Implement a **collaborative filtering** system based on historical user-game interaction patterns.
- Develop a **content-based** system that recommends games according to their attributes (genre, developer, tags, etc.).
- Incorporate a **sentiment analysis** module on user reviews using NLP.
- Build a **hybrid model** that combines the three previous approaches.
- Compare the performance of the hybrid model against the pure collaborative filtering model, using metrics such as recall@k and ndcg@k.

---

## 🛠️ Technologies Used

- **Python** (main language)
- **Pandas, NumPy** (data processing)
- **Matplotlib, Seaborn** (visualization)
- **Scikit-learn, NLTK** (ML and NLP)
- **KerasRS** (Keras 3, TensorFlow/Torch backends, recommender systems)
* **Recommender Systems**: Surprise, KerasRS (TensorFlow)
- **Hugging Face Transformers** (advanced NLP)
* **NLP Transformers**: Hugging Face Transformers

---
## 📁 Project Structure
## ⚙️ Setup and Installation
- `Data/`: Raw and processed data (parquet, npz, json, csv).
- `Code/`: Data processing, modeling, and evaluation scripts.
  - `01_generacion_dataset.py`: Main dataset generation and cleaning.
  - `02_analisis_exploratorio.py`: Exploratory data analysis.
  - `03_models/`: Recommender models and utilities.
    - `data_prep_to_npz.py`: Prepares data splits for training.

    - `utils_metrics.py`: Evaluation metrics (recall@k, ndcg@k).
    - *(coming soon)*: Content-based, sentiment analysis, and hybrid models.
- `Results/`: Experiment results (weights and metrics for each run, organized by model and timestamp).
- `Doc/`: Documentation and thesis PDF.
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
## ⚙️ Installation and Usage

1. **Clone the repository:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Create and activate a virtual environment:**

## ▶️ How to Run

   # or

1.  **Data Preparation**:
3. **Install dependencies:**
    ```sh
    python Code/01_generacion_dataset.py
    ```

2.  **Analysis and Modeling**:
    Once the processed dataset exists, you can run the analysis or modeling scripts. These scripts load the `.parquet` file directly, making them fast and efficient.
## ▶️ Running the Models
    # Example for running Exploratory Data Analysis
### 1. Data Preparation
Process raw data and generate the main dataset:

---

Prepare the train/validation/test splits in `.npz` format:

* **Matias Gabriel Flores**
