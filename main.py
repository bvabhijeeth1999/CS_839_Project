import sys
from tokenizers.tik_token import get_tokens_tik_token
from tokenizers.n_gram import get_tokens_n_gram
from string_matching.tfidf import calculate_tfidf
from string_matching.jaccard import calculate_jaccard_similarity
import pandas as pd
from pprint import pprint

def read_headlines_from_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None)
    
    # Extract headlines from the first column
    headlines = df.iloc[:, 0].tolist()
    
    return headlines

def decreasing_order(a):
    return -a[1]

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python main.py <csv file> <target headline csv file> <SLO>")
        return
    
    # Extract command-line arguments
    csv_file = sys.argv[1]
    target_file = sys.argv[2]
    slo = sys.argv[3]

    list_sentences = read_headlines_from_csv(csv_file)
    target_headlines = read_headlines_from_csv(target_file)
    target_headline = target_headlines[0]
    
    # Sample run command : 
    # python3 main.py input_list_sentences.csv target_sentence.csv 3
    
    print("-----------------------TFIDF--------------------------------")
    result1 = calculate_tfidf(target_headline, list_sentences, get_tokens_tik_token)
    result1.sort(key = decreasing_order)
    pprint(result1)

    print("-----------------------Jaccard--------------------------------")
    result2 = calculate_jaccard_similarity(target_headline, list_sentences, get_tokens_n_gram(1))
    result2.sort(key = decreasing_order)
    pprint(result2)
    
    # now once we have the models, we need to iterate through those and match with slo requirements and trigger accordingly.

    # We will iterate through all headlines in the csv file and get an array of similarity scores for story for each headline.
    # Then we will do the same for the target headline.
    # Then we will compare these arrays using some distance function and return top 5/10 or something.

    # Tokenizers (function name : get_tokens) : N-Gram, TikToken, No Tokenizer
    # String Matching Algorithms : TF-IDF, Jaccard, Edit-Distance (optional), ML (maybe after checkin)

    # Tokenizers Input : list of sentences + target sentence.
    # Tokenizers return : Sentence, list of tokens : for each sentence. (list of tokens is empty if no tokenizer)

    # String Matching Algorithms Input : <Sentence, list of tokens> for list of sentences, <Sentence, list of tokens> for target sentence.
    # String Matching Algo. O/P : Top k sentences from the input param 1, that are similar to target sentence (param2) 
    # SM intermediate function (if needed) : i/p will be sentence, (score or score with %). Now, compare the target sentence score with 
    # all other scores and return top k.

    # After that, profile all combinations (time per sentence is considered) and trigger according to user requirement from main.


if __name__ == "__main__":
    main()
