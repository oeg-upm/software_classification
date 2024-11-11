import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

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

def isWebpage(types, languages):
    if 'markup' in types and 'HTML' in languages:
        return True
    return False

# Function to make predictions on a text
def predict(text):
    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # Adjust this based on your training setup
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move input tensors to the same device as the model
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Perform the forward pass and get the logits
    with torch.no_grad():  # No need to calculate gradients for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Apply sigmoid to get probabilities for each label
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Apply a threshold of 0.5 to predict categories
    predicted_categories = (probabilities > 0.5).astype(int)

    # Print the prediction results
    print(f"Input Text: {text}")
    for i, label in enumerate(categories):
        print(f"{label}: {'Present' if predicted_categories[i] == 1 else 'Absent'}")

    return predicted_categories, probabilities

categories = ["Library", "Benchmark", "Service", "Workflow", "Other"]

#Initialize the model
# Load the trained model and tokenizer 
model = DistilBertForSequenceClassification.from_pretrained('multilabel_software_classifier')
tokenizer = DistilBertTokenizer.from_pretrained('multilabel_software_classifier')

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#Exploration of CSV file
file_path = 'type_linguistics_list.csv'

#Load CSV in a Dataframe
df = pd.read_csv(file_path)

df_types_software = pd.DataFrame()

#Iterates for each row
for index, row in df.iterrows():

    print("Analyzing repo:"+str(row['github_url']))
    #Object of categories
    software_types = {"Library":0,"Benchmark":0,"Service":0,"Workflow":0,"Script":0,"Notebook":0,"Other":0}
    #Check file extensions
    if isWebpage(row['types'],row['languages']):
        software_types['Other']=1

    if isScript(row['types'],row['languages']):
        software_types['Script']=1

    if isNotebook(row['types'],row['languages']):
        software_types['Notebook']=1

    if isWorkflow(row['types'],row['languages'],row['description']):
        software_types['Workflow']=1
        
    description = row['description']
    #If it is nor empty
    if description:
        predicted_categories, probabilities = predict(description)
        #Check if the workflow has been detected with the file extension
        software_types['Library'] = predicted_categories[0]
        software_types['Benchmark'] = predicted_categories[1]
        software_types['Service'] = predicted_categories[2]
        if software_types['Workflow'] == 0:
            predicted_categories[3]
        software_types['Other'] = predicted_categories[4]
    #If no software has been detected, we assign a 1 to Other
    if software_types['Library'] and software_types['Notebook'] and software_types['Script'] and software_types['Workflow'] and software_types['Service'] and software_types['Benchmark'] and software_types['Other']:
            software_types['Other'] = 1
    df_types_software = df_types_software._append({"github_url":row['github_url'],"description":row['description'],"topics":row['topics'],"languages":row['languages'],"types":row['languages'],"types_software":software_types},ignore_index=True)

df_types_software.to_csv('software_classification.csv',index=False)
