from sentence_transformers import SentenceTransformer, util

def calculate_llm(reference_sentence, sentence_list, tokenizer):
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

    # Calculate the reference sentence embedding
    reference_embedding = model.encode(reference_sentence, convert_to_tensor=True)

    # Calculate the Sentence Embeddings 
    embeddings = model.encode(sentence_list, convert_to_tensor=True)

    sentences_result = []
    count  = 0
    for i in range(0,len(sentence_list)):
        sentences_result.append((sentence_list[i], util.pytorch_cos_sim(reference_embedding, embeddings[i])))
        count += 1
    
    # Return the similarity scores for all sentences against the reference sentence
    return sentences_result
