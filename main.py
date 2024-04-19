import sys

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python main.py <csv file> <target headline> <SLO>")
        return
    
    # Extract command-line arguments
    csv_file = sys.argv[1]
    target_headline = sys.argv[2]
    slo = sys.argv[3]
    
    # now once we have the models, we need to iterate through those and match with slo requirements and trigger accordingly.

    # We will iterate through all headlines in the csv file and get an array of similarity scores for story for each headline.
    # Then we will do the same for the target headline.
    # Then we will compare these arrays using some distance function and return top 5/10 or something.

    # Tokenizers : N-Gram, TikToken, No Tokenizer
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
