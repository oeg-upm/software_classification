import pandas as pd
import yaml
import json
import ast

if __name__ == "__main__":
    file_path = 'results/metadata_github_list.csv'
    df = pd.read_csv(file_path)

    #df_results = pd.DataFrame(columns=['topic', 'frequency'])
    topics_dictionary={}
    count = 0
    for index, row in df.iterrows():
        print("Analyzing repo:"+str(row['github_url']))
        try:
            count += 1
            print(row['topics'])
            # Convert the string to a list
            topics_list = ast.literal_eval(row['topics'])
            for topic in topics_list:
                if topic in topics_dictionary:
                    topics_dictionary[topic] = topics_dictionary[topic] + 1
                else:
                    topics_dictionary[topic] = 1
        except Exception as ex:
            print("Exception")


    topics_dictionary = {k:[v] for k,v in topics_dictionary.items()} 
    df_results = pd.DataFrame(list(topics_dictionary.items()), columns=['topic', 'frequency'])
    #sorted_df = df_results.sort_values(by='frequency', ascending=False)
    df_results.to_csv('results/topics_distribution.csv',index=False)
