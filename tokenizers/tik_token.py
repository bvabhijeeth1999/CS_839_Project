import tiktoken

def get_tokens_tik_token(sentence):
    encoding = tiktoken.get_encoding("cl100k_base")

    token_list = encoding.encode(sentence)
    tokens = [encoding.decode_single_token_bytes(token).decode('utf-8') for token in token_list]
    
    return tokens