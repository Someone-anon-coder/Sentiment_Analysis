# Sentiment Analysis Model

## Overview

This project is a sentiment analysis model that classifies natural language input as either positive or negative. The model is trained on a labeled dataset and uses a Multinomial Naive Bayes classifier. The process involves several stages, including data cleaning, preprocessing, tokenization, vectorization, hyperparameter tuning, model training, and testing.

## Project Structure

The project consists of the following Python scripts:

1. **Data Cleaning** (`clean_dataset.py`): Cleans the dataset by filtering out neutral sentiments and relabeling positive sentiments.
2. **Data Preprocessing** (`preprocess_dataset.py`): Preprocesses the text data by removing URLs, punctuation, and stopwords, and then tokenizing the text.
3. **Data Splitting** (`split_data.py`): Splits the dataset into training and testing sets.
4. **Vectorization** (`vectorize_and_save.py`): Tokenizes and vectorizes the text data using either TF-IDF or Count Vectorizer.
5. **Hyperparameter Tuning** (`tune_hyperparameters.py`): Performs hyperparameter tuning using GridSearchCV to find the best parameters for the model.
6. **Model Training** (`train_model.py`): Trains the Multinomial Naive Bayes model and evaluates its performance.
7. **Model Testing** (`test_model.py`): Allows users to input sentences and get sentiment predictions in real-time.

## Setup and Execution

### 1. Data Cleaning

Use the `clean_dataset.py` script to clean the dataset. This script filters out neutral sentiments and relabels positive sentiments.

```python
python clean_dataset.py
```

### 2. Data Preprocessing

The `preprocess_dataset.py` script preprocesses the cleaned data by removing unnecessary elements like URLs and stopwords.

```python
python preprocess_dataset.py
```

### 3. Data Splitting

Use the `split_data.py` script to split the preprocessed data into training and testing datasets.

```python
python split_data.py
```

### 4. Tokenization and Vectorization

Tokenize and vectorize the split data using `vectorize_and_save.py`. The script supports both TF-IDF and Count Vectorizer.

```python
python vectorize_and_save.py
```

### 5. Hyperparameter Tuning

Use the `tune_hyperparameters.py` script to find the best parameters for the Multinomial Naive Bayes model.

```python
python tune_hyperparameters.py
```

### 6. Model Training

The `train_model.py` script trains the model using the best parameters found during hyperparameter tuning. It also evaluates the model's performance.

```python
python train_model.py
```

### 7. Model Testing

Test the model using the `test_model.py` script. This script allows you to input sentences and get real-time sentiment predictions.

```python
python test_model.py
```

## Model Performance

After training, the model achieved the following performance metrics:

- **Accuracy**: 0.7613
- **Precision**: 0.7711
- **Recall**: 0.7452
- **F1 Score**: 0.7580
- **Confusion Matrix**:
  ```
  [[123430  35348]
   [ 40719 119109]]
  ```

The trained model is saved as `naive_bayes_model.pkl`.

## Real-Time Sentiment Prediction

You can input sentences for real-time sentiment analysis using the `test_model.py` script. The model predicts the sentiment in less than half a second. Example predictions include:

- **Input**: "You are quite funny, you know that"
  - **Predicted Sentiment**: Positive
- **Input**: "Stop that, this instant"
  - **Predicted Sentiment**: Negative

## Dependencies

Ensure the following Python libraries are installed:

- `pandas`
- `scikit-learn`
- `nltk`
- `pickle`

You can install them using `pip`:

```bash
pip install pandas scikit-learn nltk
```

## Conclusion

This sentiment analysis model provides a robust method for classifying text as positive or negative, making it a valuable tool for various applications such as customer feedback analysis, social media monitoring, and more. The modular structure of the project allows for easy customization and extension.
