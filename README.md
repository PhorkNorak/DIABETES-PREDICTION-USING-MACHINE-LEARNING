# Taipei Diabetes Prediction

A machine learning project for predicting diabetes risk using medical data from Taipei Municipal Medical Center. This project includes exploratory data analysis (EDA), model training and comparison, and a web-based prediction app built with Streamlit.

**Disclaimer**: This is a student course project for educational purposes only. It is not intended for medical diagnosis, treatment, or clinical use. The predictions are for learning and demonstration purposes. Always consult healthcare professionals for medical advice. No ownership is claimed over the dataset or methodologies; this is a study implementation.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Resources](#project-resources)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features
- **Comprehensive EDA**: Data cleaning, missing value handling, feature distributions, and correlation analysis
- **Model Comparison**: Cross-validation comparison of 4 ML models (Logistic Regression, Random Forest, Gradient Boosting, KNN)
- **Web App**: Interactive Streamlit application for real-time diabetes risk prediction
- **Automated Pipeline**: Preprocessing (imputation, scaling) + classifier in a single pipeline
- **Model Persistence**: Trained model saved for easy deployment

## Dataset
- **Source**: Taipei Municipal Medical Center (2018-2022)
- **Size**: 15,000 female patients
- **Features**: 8 numeric features including Pregnancies, Plasma Glucose, BMI, Age, etc.
- **Target**: Diabetic (1 = diabetes diagnosed, 0 = no diabetes)
- **File**: `TAIPEI_diabetes.csv`

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/taipei-diabetes-prediction.git
   cd taipei-diabetes-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
   ```

3. Ensure the dataset file `TAIPEI_diabetes.csv` is in the root directory.

## Usage

### Training the Model
1. Open `main.ipynb` in Jupyter Notebook or VS Code
2. Run all cells to:
   - Load and clean data
   - Perform EDA
   - Train and compare models
   - Save the best model to `artifacts/best_diabetes_model.joblib`

### Running the Web App
1. After training, run:
   ```bash
   streamlit run app.py
   ```
2. Open the provided local URL in your browser
3. Enter patient data and get diabetes risk prediction

### Making Predictions Programmatically
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('artifacts/best_diabetes_model.joblib')

# Prepare input data
input_data = pd.DataFrame([{
    'Pregnancies': 2,
    'PlasmaGlucose': 120,
    'DiastolicBloodPressure': 70,
    'TricepsThickness': 25,
    'SerumInsulin': 80,
    'BMI': 26.0,
    'DiabetesPedigree': 0.5,
    'Age': 35
}])

# Get prediction
probability = model.predict_proba(input_data)[0, 1]
prediction = model.predict(input_data)[0]
print(f"Diabetes probability: {probability:.2%}")
print(f"Prediction: {'Diabetic' if prediction == 1 else 'Not diabetic'}")
```

## Model Details
- **Algorithm**: Gradient Boosting Classifier (selected via cross-validation)
- **Preprocessing**: Median imputation for missing values + Standard scaling
- **Performance**: ~95% accuracy on test set
- **Compared Models**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors

## Project Resources
Here are resources that may be helpful for the project:
- **Original study for the project dataset**: Chou CY, Hsu DY, Chou CH. Predicting the Onset of Diabetes with Machine Learning Methods. J Pers Med. 2023 Feb 24;13(3):406. doi:10.3390/jpm13030406. PMID: 36983587; PMCID: PMC10057336.
- **PIMA Indian dataset**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **WHO Diabetes webpage**: https://www.who.int/news-room/fact-sheets/detail/diabetes

## Project Structure
```
taipei-diabetes-prediction/
│
├── main.ipynb                    # Main notebook with EDA, training, and evaluation
├── EDA_diabetes.ipynb           # Additional EDA notebook
├── app.py                       # Streamlit web application
├── TAIPEI_diabetes.csv          # Dataset (not included in repo)
├── artifacts/
│   └── best_diabetes_model.joblib  # Trained model
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments
- Dataset provided by Taipei Municipal Medical Center
- Built for educational purposes as part of ML clustering project