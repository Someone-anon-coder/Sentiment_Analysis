import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize_and_save(train_df: pd.DataFrame, test_df: pd.DataFrame, vectorizer_type: str='tfidf') -> None:
    """Tokenize and Vectorize the data


    Args:
        train_df (pd.DataFrame): Training data to be vectorized
        test_df (pd.DataFrame): Test data to be vectorized
        vectorizer_type (str, optional): Vectorizer type to be used. Defaults to 'tfidf'.

    Raises:
        ValueError: If vectorizer_type is not 'tfidf' or 'count'.

    Returns:
        None
    """
    
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
    elif vectorizer_type == 'count':
        vectorizer = CountVectorizer(stop_words='english')
    else:
        raise ValueError("vectorizer_type must be 'tfidf' or 'count'")
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    with open(f'Pkl_Files/{vectorizer_type}_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('Pkl_Files/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    
    with open('Pkl_Files/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    
    print(f"Vectorizer and tokenized data saved as '{vectorizer_type}_vectorizer.pkl', 'X_train.pkl', and 'X_test.pkl'.")

if __name__ == "__main__":
    train_df = pd.read_csv('New_Data/train_split.csv')
    test_df = pd.read_csv('New_Data/test_split.csv')
    
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    
    vectorize_and_save(train_df, test_df, vectorizer_type='tfidf')
    