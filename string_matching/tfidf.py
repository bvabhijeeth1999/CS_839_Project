from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize

# Custom tokenizer function using NLTK's word_tokenize
def custom_tokenizer(sentence):
    return word_tokenize(sentence)

def calculate_tfidf(reference_sentence, sentence_list):
    # Combine the reference sentence and the list of sentences
    all_sentences = [reference_sentence] + sentence_list
    
    # Initialize the TfidfVectorizer with custom tokenizer
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
    
    # Fit the vectorizer and transform the sentences into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    
    # Get the TF-IDF scores
    tfidf_scores = (tfidf_matrix * tfidf_matrix.T).toarray()
    
    # Return the TF-IDF scores for all sentences against the reference sentence
    return tfidf_scores[0][1:]

# Example usage
reference_sentence = "This is a reference sentence."
sentence_list = [
    "This is the first sentence.",
    "This is the second sentence.",
    "This is the third sentence."
]

tfidf_scores = calculate_tfidf(reference_sentence, sentence_list)
print("TF-IDF scores:", tfidf_scores)