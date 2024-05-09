import sys
from tokenizer.tik_token import get_tokens_tik_token
from tokenizer.n_gram import get_tokens_n_gram
from tokenizer.whitespace import get_tokens_whitespace
from string_matching.tfidf import calculate_tfidf
from string_matching.jaccard import calculate_jaccard_similarity
from string_matching.llm import calculate_llm
from clean_data import clean_strings
import pandas as pd
from pprint import pprint
import time

def get_relevant_headlines(results, threshold):
        k = 20 # top 10 headlines
        t = 0
        for result in results:
            if result[1] >= threshold:
                t += 1 
        
        if(t > k):
            print("printing t records")
            return results[0:t]
            #print(t)
        else:
            print("printing k records")
            return results[0:k]
            #print(k)

def calculate_precision(tp, fp):
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    return tp / (tp + fn)

def calculate_f1_score(tp, fp, fn):
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    return 2 * (precision * recall) / (precision + recall)


def read_headlines_from_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # print(df.shape)
    # pprint(df.iloc[:,1:2])
    # Extract headlines from the first column
    headlines = df['TITLE'].to_list()
    
    #pprint(headlines)

    return headlines

def decreasing_order(a):
    return -a[1]

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 6:
        print("Usage: python main.py <csv file> <target headline csv file> <Time in micro sec> <Precision %> <Recall %>")
        return

    # Extract command-line arguments
    csv_file = sys.argv[1]
    target_file = sys.argv[2]
    target_time = float(sys.argv[3])
    target_precision = float(sys.argv[4])
    target_recall = float(sys.argv[5])

    # # Script to clean raw file
    # raw_file = csv_file  # Replace with your input CSV file path
    # csv_file = 'cleaned_data.csv'  # Replace with your output CSV file path

    # clean_strings(raw_file, csv_file)

    cleaned_data_df = pd.read_csv(csv_file)
    list_sentences = read_headlines_from_csv(csv_file)[0:10000]
    target_headlines = read_headlines_from_csv(target_file)
    target_headline = target_headlines[0]

    # Read the slo file and choose which combination fits best
    slo_df = pd.read_csv('slo.csv')
    num_sentences = len(cleaned_data_df)+1
    satisfied_models = []
    final_model_index = -1
    for index, row in slo_df.iterrows():
    # Access row values using row['column_name'] or row[index]
        computed_time = float(row['Time'])*num_sentences
        computed_precision = float(row['Precision'])*100
        computed_recall = float(row['Recall'])*100
        if(computed_time <= target_time and computed_precision >= target_precision and computed_recall >= target_recall):
            print(row['Tokenizer'])
            print(row['String Matching Algorithm'])
            satisfied_models.append(index)
    
    print(satisfied_models)
    if(len(satisfied_models) == 0):
        final_model_index = 2
    else:
        final_model_index = -1
        f1_max = 0
        for model in satisfied_models:
            if (float(slo_df.iloc[model]['F1']) > f1_max):
                f1_max = float(slo_df.iloc[model]['F1'])
                final_model_index = model

    # by here we will have our final model in final_model_index
    final_tokenizer = slo_df.iloc[final_model_index]['Tokenizer']
    final_sm_algo = slo_df.iloc[final_model_index]['String Matching Algorithm']
    final_threshold = slo_df.iloc[final_model_index]['Threshold']

    print(final_tokenizer)
    print(final_sm_algo) 

    sm_dict = {"tfidf": calculate_tfidf,"jaccard": calculate_jaccard_similarity,"llm": calculate_llm}
    tokenizer_dict = {"tik_token": get_tokens_tik_token, "n_gram": get_tokens_n_gram, "whitespace": get_tokens_whitespace}

    result = sm_dict[final_sm_algo](target_headline, list_sentences, tokenizer_dict[final_tokenizer])
    result.sort(key = decreasing_order)

    relevant_headlines = get_relevant_headlines(result, final_threshold)
    pprint(relevant_headlines)

    # # Record the start time
    # cumm_time = 0
    # for i in range(20):
    #     start_time = time.time()

    #     # Your Python code goes here
    #     # For example:
    #     result1 = calculate_llm(target_headline, list_sentences, get_tokens_tik_token)
    #     result1.sort(key = decreasing_order)

    #     # Record the end time
    #     end_time = time.time()

    #     # Calculate the elapsed time
    #     elapsed_time = end_time - start_time
    #     print(elapsed_time)
    #     cumm_time += (elapsed_time)

    # avg_time = cumm_time/(20*len(list_sentences))
    # print(avg_time)
            
    # # code to get random target sentence from cleaned df
    # target_sentences_random = []
    # main_data_path = 'cleaned_data.csv'
    # df = pd.read_csv(main_data_path)
    # df = df.iloc[:, [1, 5]]

    # def select_random_row(group):
    #     return group.sample(1, random_state=42)  # Set random_state for reproducibility

    # # Apply the function to each group of story_id
    # random_target_headlines = df.groupby('STORY', group_keys=False, sort=False).apply(select_random_row)

    # # Display the result
    # print(random_target_headlines.head(138))
    # # 138 unique stories in first 10k headings.

    # #code to get f1,p,r metrics for 10k headlines
    # f1 = 0
    # p = 0
    # r = 0
    # cnt = 0
    # for index, row in random_target_headlines.iterrows():
    #     fn = 0
    #     fp = 0
    #     tp = 0
    #     tn = 0
    #     random_target_headline = row[0]
    #     random_target_story = row[1]
    #     print("Printing random target headline")
    #     print(random_target_headline)
    #     result2 = calculate_llm(random_target_headline, list_sentences[0:10000], get_tokens_n_gram(1))
    #     result2.sort(key = decreasing_order)
    #     #print_relevant_headlines(result2,0.15)
    #     relevant_headlines = get_relevant_headlines(result2,0.45)
    #     print("Printing relevant headlines length")
    #     print(len(relevant_headlines))

    #     # search in cleaned_data_df for value result2[i]
    #     for relevant_headline in relevant_headlines:
    #         # print("printing relevant healine")
    #         # print(relevant_headline[0])
    #         relevant_headline_row = cleaned_data_df[cleaned_data_df['TITLE'] == relevant_headline[0]]
    #         # print("Printing relevant headline rows")
    #         # print(relevant_headline_row['STORY'].values[0])
    #         if(relevant_headline_row['STORY'].values[0] == random_target_story):
    #             tp += 1
    #         else:
    #             fp += 1
            
    #     relevant_stories = cleaned_data_df[cleaned_data_df['STORY'] == random_target_story]
    #     print(len(relevant_stories))

    #     fn = len(relevant_stories) - tp

    #     print(tp, tn, fp, fn)

    #     print("precision : ")
    #     p += calculate_precision(tp,fp)
    #     print(calculate_precision(tp,fp))
    #     print("recall : ")
    #     r += calculate_recall(tp,fn)
    #     print(calculate_recall(tp,fn))
    #     print("F1 : ")
    #     f1 += calculate_f1_score(tp,fp,fn)
    #     print(calculate_f1_score(tp,fp,fn))
    #     print("\n")
    
    #     cnt += 1

    #     if(cnt == 138):
    #         break
        
    # avg_precision = p/138
    # avg_recall = r/138
    # avg_f1 = f1/138

    # print("avg_precision")
    # print(avg_precision)
    # print("avg_recall")
    # print(avg_recall)
    # print("avg_f1")
    # print(avg_f1)

    

if __name__ == "__main__":
    main()
