import pandas as pd
import ast

def isScript(types, languages):
    if 'programming' in types and ('Shell' in languages or 'PowerShell' in languages, 'Batchfile' in languages):
        return True
    return False

def isWorkflow(types, languages, description):
    if 'programming' in types and ('Common Workflow Language' in languages or 'HCL' in languages or 'WDL' in languages):
        return True
    if 'workflow' in description or 'pipeline' in description:
        return True
    return False

def isNotebook(types, languages):
    if 'markup' in types and 'Jupyter Notebook' in languages:
        return True
    return False

def isService(types, languages, description):
    if 'service' in description or 'api' in description:
        return True
    return False

def isLibrary(types, description):
    try:
        if 'programming' in types or  'library' in description or 'package' in description:
            return True
    except Exception as ex:
        print("Exception")
    return False

def isBenchmark(types, description):
    if 'benchmark' in description:
        return True
    return False

if __name__ == "__main__":
    file_path = 'software_classification_v2.csv'

    num_library = 0
    num_benchmark = 0
    num_service = 0
    num_workflow = 0
    num_other = 0

    df = pd.read_csv(file_path)

    df_corpus = pd.DataFrame()
    

    for index, row in df.iterrows():
        entry_corpus = False
        print("Analyzing repo:"+str(row['github_url']))

        row['types_software']
        
        topics_list = ast.literal_eval(row['topics'])
        type_software = ast.literal_eval(row['types_software'])

        

        if 'library' in topics_list or 'package' in topics_list or 'nlp-library' in topics_list or 'python-library' in topics_list:
            num_library += 1
            if num_library <= 51:
                entry_corpus = True

        if 'benchmark' in topics_list or 'benchmarking' in topics_list:
            num_benchmark += 1
            if num_benchmark <= 51 :
                entry_corpus = True

        if 'api' in topics_list or 'rest-api' in topics_list:
            num_service += 1
            if num_service <= 51:
                entry_corpus = True

        if 'workflow' in topics_list or 'pipeline' in topics_list:
            num_workflow += 1
            if num_workflow <= 51:
                entry_corpus = True

        if not entry_corpus:
            num_other += 1
            if num_other <= 51:
                entry_corpus = True
        
        
        if entry_corpus:
            df_corpus = df_corpus._append({"description":row['description'],"Library":int(type_software['Library']),"Benchmark":int(type_software['Benchmark']),"Service":int(type_software['Service']),"Workflow":int(type_software['Workflow']),"Other":int(type_software['Other'])},ignore_index=True)
        if num_library > 50 and num_benchmark > 50 and num_other > 50 and num_workflow > 50 and num_service > 50:
            break
    print(str(num_library)+"-"+str(num_benchmark)+"-"+str(num_service)+"-"+str(num_workflow)+"-"+str(num_other))
    df_corpus.to_csv('corpus_without_script_notebook.csv',index=False)
