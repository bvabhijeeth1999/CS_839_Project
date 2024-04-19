import sys
from tokenizers.tik_token import get_tokens_tik_token
from tokenizers.n_gram import get_tokens_n_gram
import pandas as pd

def read_headlines_from_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None)
    
    # Extract headlines from the first column
    headlines = df.iloc[:, 0].tolist()
    
    return headlines


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

    for sentence in list_sentences:
        print(sentence)

    print("Printing target headline")
    print(target_headline)
    
    # Sample run command : 
    # python3 main.py input_list_sentences.csv target_sentence.csv 3

    # sample list_sentences = [
    #     "Fed official says weak data caused by weather, should not slow taper",
    #     "Fed's Charles Plosser sees high bar for change in pace of tapering",
    #     "US open: Stocks fall after Fed official hints at accelerated tapering",
    #     "Fed risks falling 'behind the curve', Charles Plosser says",
    #     "Fed's Plosser: Nasty Weather Has Curbed Job Growth",
    #     "Plosser: Fed May Have to Accelerate Tapering Pace",
    #     "Fed's Plosser: Taper pace may be too slow",
    #     "Fed's Plosser expects US unemployment to fall to 6.2% by the end of 2014",
    #     "US jobs growth last month hit by weather:Fed President Charles Plosser",
    #     "ECB unlikely to end sterilisation of SMP purchases - traders" ,
    #     "ECB unlikely to end sterilization of SMP purchases: traders",
    #     "EU's half-baked bank union could work",
    #     "Europe reaches crunch point on banking union",
    #     "ECB FOCUS-Stronger euro drowns out ECB's message to keep rates low for  ...",
    #     "EU aims for deal on tackling failing banks"
    # ]


    result_tik_token = []
    # calling tik token tokenizer.
    print("Calling tik token tokenizer")
    result_tik_token = get_tokens_tik_token(list_sentences, target_headline)

    result_n_gram = []
    # calling n_gram
    result_n_gram = get_tokens_n_gram(list_sentences, target_headline)
    
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
