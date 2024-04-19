import nltk

def get_tokens_n_gram(list_sentences, target_sentence):

    output_list_sentences = []
    output_target_sentence = []
    # Iterate through the list of sentences and generate tokens for each sentence.
    for sentence in list_sentences:
        tokens = sentence.split()
        trigrams = nltk.ngrams(tokens, 3)
        trigram_tokens = []
        for trigram in trigrams:
            merged_trigram = ' '.join(trigram)
            trigram_tokens.append(merged_trigram)
        output_list_sentences.append((sentence,list(trigram_tokens)))

    
    tokens = target_sentence.split()
    trigrams = nltk.ngrams(tokens, 3)
    trigram_tokens = []
    for trigram in trigrams:
        merged_trigram = ' '.join(trigram)
        trigram_tokens.append(merged_trigram)
    output_target_sentence.append((target_sentence,list(trigram_tokens)))
    
    print("\nprinting list of sentences with tokens from n_gram\n")
    for tup in output_list_sentences:
        print(tup)

    print("\nPrinting target sentence tokens\n")
    for tup in output_target_sentence:
        print(tup)
    
    return (output_list_sentences, output_target_sentence)