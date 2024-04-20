import pandas as pd
import tiktoken

def clean_strings(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Iterate over each column and row to check for problematic strings
    for col in df.columns:
        for idx, value in df[col].items():
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                token_list = encoding.encode(value)
                tokens = [encoding.decode_single_token_bytes(token).decode('utf-8') for token in token_list]
            except Exception:
                print("inside exception " + str(idx))
                # Remove problematic string
                df.at[idx, col] = ''

    # Write the cleaned DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

    print("Problematic strings removed successfully.")


