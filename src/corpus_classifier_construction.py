import pandas as pd
import ast

sample_limit = 50

if __name__ == "__main__":
    file_path = 'programming_list.csv'

    df = pd.read_csv(file_path)

    df_corpus = pd.DataFrame()

    end_process = False

    num_library = 0
    num_benchmark = 0
    num_service = 0
    num_workflow = 0
    num_other = 0

    for index, row in df.iterrows():
        library_detected = False
        benchmark_detected = False
        service_detected = False
        workflow_detected = False
        other_detected = False
        detection = False

        print("Analyzing repo:"+str(row['github_url']))

        try:
            topics_list = ast.literal_eval(row['topics'])
            if 'library' in topics_list or 'package' in topics_list or 'nlp-library' in topics_list or 'python-library' in topics_list:
                library_detected = True
                detection = True
                num_library += 1

            if 'benchmark' in topics_list or 'benchmarking' in topics_list:
                benchmark_detected = True
                detection = True
                num_benchmark += 1

            if 'api' in topics_list or 'rest-api' in topics_list:
                service_detected = True
                detection = True
                num_service += 1

            if 'workflow' in topics_list or 'pipeline' in topics_list:
                workflow_detected = True
                detection = True
                num_workflow += 1

            if topics_list and not library_detected and not benchmark_detected and not service_detected and not workflow_detected:
                other_detected = True
                detection = True
                num_other += 1

            if detection:
                if num_library <= sample_limit and num_benchmark <= sample_limit and num_service <= sample_limit and num_workflow <= sample_limit and num_other <= sample_limit:
                    df_corpus = df_corpus._append({"github_url":row["github_url"],"description":row['description'],"Library":1 if library_detected else 0,"Benchmark":1 if benchmark_detected else 0,"Service":1 if service_detected else 0,"Workflow":1 if workflow_detected else 0,"Other":1 if other_detected else 0},ignore_index=True)
                else:
                    if library_detected:
                        num_library = num_library - 1
                    if benchmark_detected:
                        num_benchmark = num_benchmark - 1
                    if service_detected:
                        num_service = num_service - 1
                    if workflow_detected:
                        num_workflow = num_workflow - 1
                    if other_detected:
                        num_other = num_other - 1

            if num_library == 50 and num_benchmark == 50 and num_service == 50 and num_workflow == 50 and num_other == 50:
                break

        except Exception as ex:
            print("Exception")

    print(str(num_library)+"-"+str(num_benchmark)+"-"+str(num_service)+"-"+str(num_workflow)+"-"+str(num_other))
    df_corpus.to_csv('corpus_to_annotate.csv',index=False)
    df_corpus.to_csv('corpus_for_classifier.csv',columns=['description', 'Library', 'Benchmark','Service','Workflow','Other'],index=False)
