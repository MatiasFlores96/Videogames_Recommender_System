# Hybrid Videogame Recommender System

This repository contains the source code for the Master's Thesis project **"Hybrid Videogame Recommender System Integrating User Behavior, Game Attributes, and Community Ratings"** by Matías Flores. This project is part of the Master's in Data Science program at the Instituto Tecnológico de Buenos Aires (ITBA).

The core objective is to develop and evaluate a hybrid recommendation system that improves upon traditional collaborative filtering methods by integrating multiple data sources. The system aims to address the information overload problem on videogame platforms like Steam, providing users with more accurate, relevant, and personalized recommendations.

## 🚀 Project Goals

The main goals of this thesis are:

* To implement a **collaborative filtering** model based on historical user interaction patterns.
* To develop a **content-based** model that recommends games based on their specific attributes (genre, developer, tags).
* To incorporate a **sentiment analysis** module using NLP to analyze user reviews and gauge community perception.
* To build and evaluate a **hybrid model** that combines the three approaches to enhance recommendation accuracy and relevance.
* To compare the performance of the hybrid system against a traditional collaborative filtering baseline using metrics like precision, recall, and F1-score.

---

## 🛠️ Tech Stack

This project is developed in Python and leverages the following core libraries:

* **Data Manipulation & Analysis**: Pandas, NumPy
* **Data Visualization**: Matplotlib, Seaborn
* **Machine Learning & NLP**: Scikit-learn, NLTK
* **Recommender Systems**: Surprise, KerasRS (TensorFlow)
* **Deep Learning**: TensorFlow, Keras
* **NLP Transformers**: Hugging Face Transformers

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/Videogames_Recommender_System.git](https://github.com/your-username/Videogames_Recommender_System.git)
    cd Videogames_Recommender_System
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run

The project workflow is split into two main stages:

1.  **Data Preparation**:
    First, run the data generation script. This will process the raw `.json` files from the `Data/` directory and save a clean, unified dataset as `dataset_procesado.parquet`. This step only needs to be run once.
    ```sh
    python Code/01_generacion_dataset.py
    ```

2.  **Analysis and Modeling**:
    Once the processed dataset exists, you can run the analysis or modeling scripts. These scripts load the `.parquet` file directly, making them fast and efficient.
    ```sh
    # Example for running Exploratory Data Analysis
    python Code/02_analisis_exploratorio.py
    ```

---

## ✍️ Author

* **Matias Gabriel Flores**