# Drug Prediction and Polypharmacy System
# **Developed by Sumit Kumar - 2025**
[![Project License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/downloads/)
[![Colab Notebooks](https://colab.research.google.com/assets/colab-badge.svg)](link-to-your-colab-notebook-here) <!-- Replace with your Colab Notebook Link if applicable -->

**Predict medications, assess polypharmacy risk, and predict diseases with AI.**

This repository contains the code for a Drug Prediction and Polypharmacy System, built using a Biomedical NLP model, `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`. This system is designed to assist in healthcare decision support by providing:

*   💊 **Medication Recommendations:** Intelligent suggestions for appropriate medications based on patient symptoms and medical context.
*   ⚠️ **Polypharmacy Risk Assessment:** Identification of potential risks associated with using multiple medications concurrently.
*   🩺 **Disease Prediction:** Prediction of the likely disease a patient might be suffering from, based on their presented symptoms.

**[Optional: Insert a GIF or short video demo here showing the interactive widget in action.  This significantly enhances the "interactive" feel of the README.]**

## Table of Contents

*   [Project Overview](#project-overview)
*   [Key Features](#key-features)
*   [Interactive Demo](#interactive-demo)
*   [Model Architecture](#model-architecture)
*   [Dataset](#dataset)
*   [Performance](#performance)
*   [Deployment](#deployment)
*   [Quick Start](#quick-start)
*   [Requirements](#requirements)
*   [License](#license)
*   [Contributions](#contributions)

## Project Overview

This project aims to leverage the power of Natural Language Processing (NLP) and specifically biomedical language models to create a system that can provide valuable insights for healthcare professionals and patients regarding drug prescriptions and potential polypharmacy risks.

The system takes patient information, including:

*   Age, Gender, Blood Group, Weight
*   Symptoms (and their severity)
*   (Optionally) Medical History and Allergies

...and utilizes a fine-tuned `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` model to predict:

*   **Top 3 Recommended Medications** with dosage, frequency, instructions, duration, and confidence scores.
*   **Polypharmacy Risk Level** (Low to Medium, Medium to High, Unknown).
*   **Predicted Disease**

## Key Features

*   **Biomedical NLP Model:**  Utilizes the `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` model, pre-trained on a vast corpus of biomedical text for enhanced medical text understanding.
*   **Multi-Task Learning:**  Simultaneously predicts medications, polypharmacy risk, and disease for improved efficiency and knowledge sharing.
*   **Enhanced Text Input:** Combines patient demographics, symptoms, and medical context into a structured input for richer information processing.
*   **Class Imbalance Handling:**  Implements weighted loss functions to address class imbalance issues, particularly in medication prediction.
*   **Interactive Prediction Interface:** Includes a user-friendly widget-based interface (in the Jupyter Notebook) for easy experimentation and demonstration.
*   **Comprehensive Output:** Provides detailed predictions including medication recommendations with usage instructions, polypharmacy risk assessment, and disease insights.

## Interactive Demo

**[Option 1: Link to Colab Notebook (Highly Recommended for "Interactive")]**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uiNrRBHW6t-8p32Xq7FVf1cfL-TChG7_?authuser=4#scrollTo=q3aiZXxHkVUb)

> Click the "Open in Colab" badge above to launch an interactive Jupyter Notebook in Google Colab. You can run the notebook and use the interactive widget at the end to test the Drug Prediction and Polypharmacy System directly in your browser!  (Remember to replace `link-to-your-colab-notebook-here` with your actual Colab notebook link).

**[Option 2:Interactive widget in the notebook.]**
![image](https://github.com/user-attachments/assets/0f624018-82cb-4cb2-9a7f-d26b1cc4d7a3)


>  To experience the interactive prediction interface, please run the Jupyter Notebook (`Drug_Prediction_and_Polypharmacy_System5.ipynb`). Cell 20 contains a widget-based form where you can input patient information and get real-time predictions from the model.

**[Optional:  If you have a deployed web app, link to it here as Option 3]**

> **[Option 3: Try the Web App (if deployed)]**
>
>  [Link to your deployed web application]
>
>  You can also access a deployed version of the system as a web application [at the provided link].  This allows you to test the system without running any code locally.

## Model Architecture

The system is built upon the `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` pre-trained model.  The architecture is enhanced with:

*   **Multi-task heads:** Task-specific layers for medication prediction (multi-label), polypharmacy risk (multi-class), and disease prediction (multi-class).
*   **Common Representation Layer:** A shared dense layer to facilitate knowledge transfer between tasks.
*   **Dropout and Weight Initialization:**  Regularization and weight initialization techniques are used for improved model performance and training stability.

**[Optional: Consider adding a simple diagram of the model architecture here.]**

## Dataset

The model was trained and evaluated on a large patient dataset consisting of **15,000 patient records**.  The dataset includes features such as:

*   Patient demographics (Age, Gender, Blood Group, Weight)
*   Symptoms and Severity Scores
*   Medical History and Allergies
*   Prescribed Medications (Medicine_1, Medicine_2, Medicine_3 with dosage, frequency, instruction, duration)
*   Polypharmacy Risk Level
*   Predicted Disease and related information (causes, prevention, health tips)

**[Optional: You could add a link to download a sample dataset or describe data sources (if applicable and permissible).]**

## Performance

The model's performance on the test dataset is summarized below:

*   **Medication Prediction:**
    *   Accuracy: 0.9377
    *   F1-score: 0.9480
*   **Polypharmacy Risk Prediction:**
    *   Accuracy: 0.9960
    *   F1-score: 0.9960
*   **Disease Prediction:**
    *   Accuracy: 0.9377
    *   F1-score: 0.9187

**[Optional: You can include a link to the `training_history.png` plot from the notebook to visually represent training progress.]**

**Note:**  Medication prediction is a multi-label task, and accuracy is measured as the set match of predicted vs. actual medications. F1-scores are weighted averages.  Disease and Polypharmacy risk are multi-class classification tasks.

## Deployment

The repository includes all the necessary artifacts for deploying the model:

*   **`BiomedNLP_drug_prediction_model_full.pt`**: Contains the full trained model (weights, configuration, class mappings).
*   **`label_encoders.pkl`**: Saved `MultiLabelBinarizer` and `LabelEncoder` instances for label transformations.
*   **`requirements.txt`**: Lists Python package dependencies.

You can deploy the model as a:

*   **Web Application:**  Using frameworks like Flask or FastAPI to create a REST API for predictions.
*   **Cloud-based Service:** Deploying the model on platforms like AWS, Google Cloud, or Azure for scalable access.
*   **Local Application:** Integrating the model into desktop or mobile applications.

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[Your GitHub Username]/[Repository Name].git
    cd [Repository Name]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the model artifacts (if not already in the repo - you may want to include the `mobileBERT_drug_prediction_model_full.pt` and `label_encoders.pkl` files directly in your repository for easier setup):**
    ```bash
    # [Instructions on how to download model files if you are not including them in the repo directly]
    # For example:
    # gdown <link_to_your_model_files_on_Google_Drive_or_other_hosting>
    ```

5.  **Run the Jupyter Notebook (`Drug_Prediction_and_Polypharmacy_System.ipynb`)**:
    ```bash
    jupyter notebook Drug_Prediction_and_Polypharmacy_System.ipynb
    ```

6.  **Experiment with the interactive prediction widget** at the end of the notebook.

7.  **To use the prediction function in your own Python code:**

    ```python
    import torch
    import pickle
    from transformers import AutoTokenizer, AutoModel
    from model import EnhancedMedicationModel  # Assuming you have a model.py file

    # Load model artifacts and encoders (replace with your actual paths)
    model_artifacts = torch.load('mobileBERT_drug_prediction_model_full.pt', map_location=torch.device('cpu')) # or 'cuda'
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    model_config = model_artifacts['model_config']
    model = EnhancedMedicationModel(**model_config)
    model.load_state_dict(model_artifacts['model_state_dict'])
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    mlb = label_encoders['mlb']
    le_risk = label_encoders['le_risk']
    le_disease = label_encoders['le_disease']
    device = torch.device('cpu') # or 'cuda' if you have GPU

    # Example patient data (replace with your own)
    patient_data = {
        'Age': 65,
        'Gender': "Female",
        'Blood_Group': "A+",
        'Weight_kg': 70.5,
        'Symptoms': "Headache; Dizziness; Chest pain",
        'Severity_Scores': "Headache:3; Dizziness:2; Chest pain:4"
    }

    # Import the prediction function from your notebook (Cell 14) or put it in a separate file
    from Drug_Prediction_and_Polypharmacy_System5 import predict_full_health_profile # Assuming notebook is in the same directory

    prediction = predict_full_health_profile(patient_data, model, tokenizer, mlb, le_risk, le_disease, device)
    print(prediction) # Explore the prediction output
    ```

## Requirements

*   Python 3.x
*   Install the required Python packages using: `pip install -r requirements.txt`

```text
pandas
numpy
torch
transformers
scikit-learn
tqdm
matplotlib
seaborn
pickle
ipywidgets
