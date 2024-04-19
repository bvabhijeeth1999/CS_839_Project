import tiktoken
import sys

# load an encoding by name
encoding = tiktoken.get_encoding("cl100k_base")

# to convert string into a list of token integers.
token_list = encoding.encode(sys.argv[1])

# print(token_list)
# print(len(token_list))

tokens = [encoding.decode_single_token_bytes(token) for token in token_list]

# file_path = "tt_tokens.txt"
# with open(file_path, 'w') as file:
#     for item in tokens:
#         file.write(str(item) + '\n')

# print(tokens)