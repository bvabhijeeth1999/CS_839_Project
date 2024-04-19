from nltk.tokenize import word_tokenize

# Custom tokenizer function using NLTK's word_tokenize
def custom_tokenizer(sentence):
    return set(word_tokenize(sentence.lower()))

def jaccard_similarity(reference_sentence, sentence):
    # Tokenize the sentences
    reference_tokens = custom_tokenizer(reference_sentence)
    sentence_tokens = custom_tokenizer(sentence)
    
    # Calculate the Jaccard similarity
    intersection = len(reference_tokens.intersection(sentence_tokens))
    union = len(reference_tokens.union(sentence_tokens))
    jaccard_sim = intersection / union
    
    return jaccard_sim

def calculate_jaccard_similarity(reference_sentence, sentence_list):
    jaccard_similarities = []
    for sentence in sentence_list:
        jaccard_sim = jaccard_similarity(reference_sentence, sentence)
        jaccard_similarities.append(jaccard_sim)
    return jaccard_similarities

# Example usage
reference_sentence = "This is a reference sentence."
sentence_list = [
    "This is the first sentence.",
    "This is the second sentence.",
    "This is the third sentence."
]

jaccard_similarities = calculate_jaccard_similarity(reference_sentence, sentence_list)
print("Jaccard similarities:", jaccard_similarities)
