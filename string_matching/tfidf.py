from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf(reference_sentence, sentence_list, tokenizer):
    # Combine the reference sentence and the list of sentences
    all_sentences = [reference_sentence] + sentence_list

    # Initialize the TfidfVectorizer with custom tokenizer
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)
    
    # Fit the vectorizer and transform the sentences into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    
    # Get the TF-IDF scores
    tfidf_scores = (tfidf_matrix * tfidf_matrix.T).toarray()
    
    sentences_result = []
    count  = 0
    for sentence in sentence_list:
        sentences_result.append((sentence, tfidf_scores[0][1:][count]))
        count += 1
    
    # Return the TF-IDF scores for all sentences against the reference sentence
    return sentences_result
