import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# import nltk

# nltk.download('stopwords')
# nltk.download('punkt')

index = 1

def preprocess_text(text: str) -> str:
    """Preprocess Data from the dataset one text at a time, like lowercasing words and removing urls

    Args:
        text (str): Text to be preprocessed

    Returns:
        str: Text after preprocessing
    """
    
    global index
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    
    text = ' '.join(words)
    print(f"Done with text {index}")
    index += 1
    
    return text

def preprocess_dataset(filename: str, new_filename: str) -> None:
    """Preprocess the dataset

    Args:
        filename (str): Dataset to be preprocessed
        new_filename (str): New dataset to be created

    Returns:
        None
    """
    
    df = pd.read_csv(filename)
    df['text'] = df['text'].apply(preprocess_text)
    
    preprocessed_file_name = f"Data/{new_filename}"
    df.to_csv(preprocessed_file_name, index=False)
    print(f"Preprocessed file saved as {preprocessed_file_name}")

if __name__ == "__main__":
    filename = "Data/train.csv"
    new_filename = "new_train_preprocessed.csv"
    
    preprocess_dataset(filename, new_filename)