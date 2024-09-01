import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_model_and_vectorizer(vectorizer_file: str, model_file: str) -> tuple:
    """Load the model and vectorizer

    Args:
        vectorizer_file (str): Vectorizer file to be loaded
        model_file (str): Model file to be loaded

    Returns:
        tuple: vectorizer, model
    """
    
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    return vectorizer, model

def preprocess_text(text: str, vectorizer: object) -> object:
    """Preprocess the text

    Args:
        text (str): Text to be preprocessed
        vectorizer (object): Vectorizer object

    Returns:
        object: Preprocessed text vector
    """
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]

    text = ' '.join(words)

    text_vector = vectorizer.transform([text])    
    return text_vector

def test_model(vectorizer_file: str, model_file: str) -> None:
    """Test the model
    
    Args:
        None

    Returns:
        None
    """
    
    vectorizer, model = load_model_and_vectorizer(vectorizer_file, model_file)
    
    while True:
        user_input = input("Enter a sentence for sentiment analysis (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break
        
        text_vector = preprocess_text(user_input, vectorizer)
        
        prediction = model.predict(text_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        print(f"Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    vectorizer_file = 'Pkl_Files/tfidf_vectorizer.pkl'
    model_file = 'Pkl_Files/naive_bayes_model.pkl'

    test_model(vectorizer_file, model_file)