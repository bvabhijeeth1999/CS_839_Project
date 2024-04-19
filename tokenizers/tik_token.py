import tiktoken

def get_tokens_tik_token(list_sentences, target_sentence):
    # load an encoding by name
    encoding = tiktoken.get_encoding("cl100k_base")

    output_list_sentences = []
    output_target_sentence = []
    # to convert string into a list of token integers.
    # Now, here need to iterate through the list of sentences and generate tokens for each sentence.
    for sentence in list_sentences:
        token_list = encoding.encode(sentence)
        tokens = [encoding.decode_single_token_bytes(token).decode('utf-8') for token in token_list]
        output_list_sentences.append((sentence,tokens))

    
    token_list = encoding.encode(target_sentence)
    tokens = [encoding.decode_single_token_bytes(token).decode('utf-8') for token in token_list]
    output_target_sentence.append((sentence,tokens))
    
    print("\nprinting list of sentences with tokens from tiktoken\n")
    for tup in output_list_sentences:
        print(tup)

    print("\nPrinting target sentence tokens\n")
    for tup in output_target_sentence:
        print(tup)
    
    return (output_list_sentences, output_target_sentence)