# Software Detection and Classification Pipeline
This repository contains the necessary scripts and partial results to reproduce the experiments of a paper presented to a conference. Name of the conference has been omitted on purpose to preserve the anonimity of the authors.
## Purpose

Pipeline to identify the presence of software in Github repositories. Also, classify the software in different categories: Workflows, Notebooks, Libraries, Services, Scripts, Benchmark and Others. 
## Description

This pipeline is composed by the following scripts:
* add_github_metadata.py: It takes a list of github repositories from a CSV file, and create a new CSV file with the following metadata fields from Github (types of language, description and topics)
* multilabel_classifier: This script classify the software in this categories: Worksflow, Benchmark, Library and Other. It has been trained from a DistilBERT model. It uses the description field used by Github. 

Another additional scripts are presented in the repository to reproduce the experiments:
* calculate_distribution: This script calculates the distribution of the categories identified in the repositories.
* calculate_topics: This script identify the topics associated to the repositories selected in our study, and calculate their frequency.
* corpus_classifier_constructor: This script builds a corpus with the categories used by the classifier. These categories are based on the topics detected. Fifty samples of each categorie were selected. This script generates a first version of the corpus. A second corpus is produced after a manual validation.
* inserData.py: This script upload to a mongodb database the OpenAire Research Graph dump for software artefacts.

### Resources

* corpora/corpus_classifier: It is the corpus used to train and evaluate the multilabel classifier.
* datasets/github_openaire.csv: This csv contains the Github repositories present in the OpenAire Research Graph.
* datasets/languages.yml: It is the database (yaml file) of the Linguistics library where specify the type of file based on its extension.
* resuts/metadata_github_list.csv: It is a CSV with the metadata extracted from Github repositories.
* results/software_classification: It is the final classification of the GIthub repositories.
* results/topcis_distribution: List of topics of the selected repositories. Frequency can be found together with the topic.