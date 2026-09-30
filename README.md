# Grab Sentiment Prototype

A machine learning prototype for analyzing **customer satisfaction and sentiment from Grab user reviews**.

The project analyzes Android and iOS reviews, classifies customer sentiment, evaluates model performance, and identifies recurring dissatisfaction factors found in negative user feedback.

## Project Overview

This prototype was developed to examine customer satisfaction using user-generated Grab reviews.

The system performs:

- Sentiment classification of customer reviews
- Comparison of multiple machine learning approaches
- Evaluation of model performance across Android and iOS reviews
- Identification of common dissatisfaction factors
- Topic analysis of negative reviews
- Language auditing and dataset cleaning
- Live sentiment prediction through a Streamlit interface

## Final Selected Model

The final sentiment classifier uses:

**TF-IDF + Support Vector Machine (SVM)**

The selected trained model is located at:

```text
models/final_cleaned/tfidf_svm_selected.joblib
```

## Project Structure

```text
GrabSentimentPrototype/
│
├── app.py
├── analyze_dissatisfaction_factors_cleaned.py
├── audit_language_contamination.py
├── create_cleaned_dataset.py
├── evaluate_platforms.py
├── evaluate_selected_model_cleaned.py
├── manual_label_validation.py
├── prepare_cleaned_splits.py
├── run_lda_dissatisfaction.py
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   └── preprocessing.py
│
├── models/
│   └── final_cleaned/
│       └── tfidf_svm_selected.joblib
│
├── data/
│   ├── android-reviews.xlsx
│   ├── ios-reviews.xlsx
│   ├── reviews_cleaned.csv
│   ├── manual-validation-results.xlsx
│   └── manual-validation-sample.xlsx
│
└── results/
    ├── all_model_comparison.csv
    ├── dissatisfaction_cleaned/
    ├── platform_evaluation_cleaned/
    └── language_audit/
```

## Sentiment Classification

The prototype classifies reviews into sentiment categories using the selected SVM model.

Several machine learning approaches were explored during model development, including:

- Naive Bayes
- Random Forest
- Support Vector Machine
- TF-IDF-based models
- Word2Vec-based models
- Multilingual BERT embeddings

The final model was selected based on evaluation results and its suitability for the prototype.

## Platform Evaluation

Model performance is evaluated using reviews from:

- Android
- iOS
- Combined Android and iOS datasets

Evaluation outputs include:

- Classification reports
- Confusion matrices
- Platform performance summaries
- Test-set predictions

These results are stored in:

```text
results/platform_evaluation_cleaned/
```

## Dissatisfaction Analysis

Negative reviews are further analyzed to identify recurring dissatisfaction factors and topics.

The analysis includes:

- Negative review extraction
- Topic modeling
- Topic distribution analysis
- Keyword-in-context analysis
- Identification of recurring dissatisfaction themes

Results are stored in:

```text
results/dissatisfaction_cleaned/
```

## Language Audit

A language audit was conducted to identify possible Indonesian or Malay reviews that could affect the intended review dataset.

The audit includes:

- Language inspection
- Identification of borderline reviews
- Manual review
- Dataset cleaning summaries
- Platform-level language contamination analysis

Results are stored in:

```text
results/language_audit/
```

## Running the Prototype

Open Terminal and navigate to the project directory:

```bash
cd GrabSentimentPrototype
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser and allow you to interact with the sentiment analysis prototype.

## Main Prototype Capabilities

The Streamlit prototype provides functionality for:

- Live review sentiment prediction
- Presentation of sentiment analysis results
- Model evaluation results
- Platform comparison
- Customer dissatisfaction analysis

## Data

The project uses Grab user reviews collected from Android and iOS sources.

The `data/` directory contains:

- Original Android and iOS review datasets
- Cleaned review data
- Manual validation samples
- Manual validation results

## Results

Generated analysis and evaluation outputs are stored in the `results/` directory.

The main result categories include:

- Overall model comparison
- Platform-level model evaluation
- Dissatisfaction and topic analysis
- Language audit and dataset-cleaning results

## Purpose

This project demonstrates how machine learning and natural language processing can be used to analyze customer feedback, evaluate sentiment, compare model performance, and identify recurring service concerns from user reviews.
