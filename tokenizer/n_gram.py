import nltk

def get_tokens_n_gram(n):
    N = n

    def n_gram(sentence):
        tokens = sentence.split()
        ngrams = nltk.ngrams(tokens, N)
        ngram_tokens = []
        for ngram in ngrams:
            merged_ngram = ' '.join(ngram)
            ngram_tokens.append(merged_ngram)
        ngram_tokens = list(ngram_tokens)
        
        return ngram_tokens

    return n_gram