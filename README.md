# Heart Disease Prediction Project

A comprehensive machine learning project for predicting heart disease using the UCI Heart Disease dataset. This project demonstrates end-to-end data science workflow including data preprocessing, feature engineering, model training, evaluation, and deployment.

## 🎯 Project Overview

This project implements multiple machine learning approaches to predict heart disease presence in patients using clinical and demographic features.

## 📊 Final Deliverables

✅ **Cleaned dataset with selected features**
- `data/heart_disease_preprocessed.csv` - Fully preprocessed dataset
- `data/heart_disease_selected_features.csv` - Optimized feature subset

✅ **Dimensionality reduction (PCA) results**
- `data/heart_disease_pca.csv` - PCA-transformed features
- Optimal components analysis and variance explanation

✅ **Trained supervised and unsupervised models**
- Supervised: Logistic Regression, Random Forest, SVM, Decision Tree
- Unsupervised: K-Means and Hierarchical Clustering

✅ **Performance evaluation metrics**
- Accuracy, Precision, Recall, F1-Score
- ROC curves and AUC analysis
- Cross-validation results

✅ **Hyperparameter optimized model**
- GridSearchCV and RandomizedSearchCV implementation
- Best model: Logistic Regression (86.7% accuracy)

✅ **Saved model in .pkl format**
- `models/model.pkl` - Production-ready trained model

✅ **GitHub repository with all source code**
- Complete project structure with organized notebooks

## 📁 Project Structure

```
Heart_Disease_Project/
├── data/                                    # Data files
│   ├── heart_disease.csv                   # Original dataset
│   ├── heart_disease_preprocessed.csv      # Cleaned data
│   ├── heart_disease_selected_features.csv # Selected features
│   └── heart_disease_pca.csv              # PCA transformed
├── notebooks/                              # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb         # Data cleaning & encoding
│   ├── 02_pca_analysis.ipynb              # Principal Component Analysis
│   ├── 03_feature_selection.ipynb         # Feature selection methods
│   ├── 04_supervised_learning.ipynb       # Classification models
│   ├── 05_unsupervised_learning.ipynb     # Clustering analysis
│   ├── 06_hyperparameter_tuning.ipynb     # Model optimization
│   └── 07_model_export.ipynb              # Model serialization
├── models/                                 # Trained models
│   └── model.pkl                          # Best performing model
├── requirements.txt                        # Dependencies
└── README.md                              # Project documentation
```

## 🚀 Quick Start

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/mohamedelziat50/Heart_Disease_Project.git
cd Heart_Disease_Project

# Create virtual environment
python -m venv heart_disease_env
heart_disease_env\Scripts\activate  # Windows
# source heart_disease_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

```

### Run the Analysis
Execute notebooks in order:
1. `01_data_preprocessing.ipynb` - Clean and prepare data
2. `02_pca_analysis.ipynb` - Dimensionality reduction
3. `03_feature_selection.ipynb` - Select optimal features
4. `04_supervised_learning.ipynb` - Train classification models
5. `05_unsupervised_learning.ipynb` - Clustering analysis
6. `06_hyperparameter_tuning.ipynb` - Optimize models
7. `07_model_export.ipynb` - Export final model

### Use the Model
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/model.pkl')

# Make predictions (requires preprocessed data)
# prediction = model.predict(X_new)
```

## 📈 Key Results

- **Best Model**: Logistic Regression
- **Accuracy**: 86.7%
- **Features**: 22 preprocessed features
- **Clustering**: K-Means identified 3 natural patient groups
- **PCA**: 95% variance retained with reduced dimensions

## 🔬 Methodology

### Data Preprocessing
- Missing value handling
- One-hot encoding for categorical variables
- Feature standardization

### Feature Engineering
- Principal Component Analysis (PCA)
- Statistical feature selection
- Correlation analysis

### Model Training
- Multiple supervised learning algorithms
- Unsupervised clustering analysis
- Cross-validation and performance evaluation

### Model Optimization
- Grid search and randomized search
- Hyperparameter tuning
- Model comparison and selection

## 📊 Dataset

**Source**: UCI Heart Disease Dataset
- **Samples**: 297 patients (after cleaning)
- **Features**: 13 clinical measurements
- **Target**: Heart disease presence (binary classification)

*This project demonstrates a complete machine learning pipeline from data preprocessing to model deployment.*