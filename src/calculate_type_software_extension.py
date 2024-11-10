import pandas as pd
import yaml
import json

if __name__ == "__main__":
    file_path = 'results/metadata_github_list.csv'
    yaml_file = 'datasets/languages.yml'

    init_position = 0
    end_position = 1
    batch_size = 10

    number_programming = 0
    number_programming_along = 0
    number_without_programming = 0
    number_only_prose = 0
    number_only_data = 0
    number_only_markdown = 0
    total_number = 0
    number_data_software = 0

    types_found = {}
    
    df_results = pd.DataFrame(columns=['github_url', 'types'])

    #Read until de first line indicates in init_position
     # Read the entire CSV file into a DataFrame
    df = pd.read_csv(file_path)

    with open(yaml_file) as stream:
        try:
            data_loaded = yaml.safe_load(stream)

            for index, row in df.iterrows():

                print("Analyzing repo:"+str(row['github_url']))
                #print("Languages:"+str(row['languages']))

                # Convert the JSON string to a dictionary
                try:
                    language_dict = json.loads(row['languages'].replace("'","\""))

                    type_software={}
                    # Iterate through the dictionary
                    software_value = False
                    no_software = False
                    prose_value = False
                    markdown_value = False
                    data_value = False
                    for language, size in language_dict.items():
                        #print ("Key:"+str(language))
                        #print ("Value:"+str(data_loaded[language]['type']))
                        if language in data_loaded:
                            type = data_loaded[language]['type']

                            if type not in types_found:
                                types_found[type] = 1
                            else:
                                types_found[type] += 1 
                            if type in type_software:
                                type_software[type]=type_software[type]+size
                            else:
                                type_software[type]=size
                            
                            if type == "programming":
                                software_value = True
                            else:
                                no_software = True

                            if type == "markdown":
                                markdown_value = True
                            if type == "prose":
                                prose_value = True
                            if type == "data":
                                data_value = True
                    
                    if software_value and not no_software:
                        number_programming_along += 1

                    if not software_value and no_software:
                        number_without_programming += 1

                    if software_value and no_software:
                        number_programming += 1

                    if markdown_value and not software_value:
                        number_only_markdown += 1

                    if prose_value and not software_value:
                        number_only_prose += 1

                    if data_value and not software_value:
                        number_only_data += 1

                    if data_value and software_value:
                        number_data_software += 1

                    total_number += 1

                    #list_types = {"github_url":row['github_url'],"description":row['description'],"types":type_software}
                    df_results = df_results._append({"github_url":row['github_url'],"description":row['description'],"topics":row['topics'],"languages":row['languages'],"types":type_software},ignore_index=True)
                except Exception as exp:
                    print("Exception")

            df_results.to_csv('results/extensions_list.csv',index=False)
             
            print(100*(number_programming_along/total_number))
            print(100*(number_without_programming/total_number))
            print(100*(number_programming/total_number))
            print(100*(number_only_markdown/total_number))
            print(100*(number_only_data/total_number))
            print(100*(number_only_prose/total_number))
            print(100*(number_data_software/total_number))
            print(total_number)
            print(types_found)
                #df_results._append({"url":row['github_url'],"types":type_software},ignore_index=True)
                #print(type_software)
                #df_results.to_csv('types_list.csv',index=False)
                #break
        except yaml.YAMLError as exc:
            print(exc)

