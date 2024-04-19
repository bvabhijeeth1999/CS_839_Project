def jaccard_similarity(reference_sentence, sentence, tokenizer):
    # Tokenize the sentences
    reference_tokens = set(tokenizer(reference_sentence))
    sentence_tokens = set(tokenizer(sentence))
    
    # Calculate the Jaccard similarity
    intersection = len(reference_tokens.intersection(sentence_tokens))
    union = len(reference_tokens.union(sentence_tokens))
    jaccard_sim = intersection / union
    
    return jaccard_sim

def calculate_jaccard_similarity(reference_sentence, sentence_list, tokenizer):
    jaccard_similarities = []

    for sentence in sentence_list:
        jaccard_sim = jaccard_similarity(reference_sentence, sentence, tokenizer)
        jaccard_similarities.append(jaccard_sim)
    
    sentences_result = []
    count  = 0
    for sentence in sentence_list:
        sentences_result.append((sentence, jaccard_similarities[count]))
        count += 1

    return sentences_result
