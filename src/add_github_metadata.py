import requests
import re
import pandas as pd
import os
import datetime
import time

def get_repo_metadata(owner, repo, token):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {'Authorization': f'token {token}'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an error for HTTP error codes
        response_headers = response.headers
        if int(response_headers['X-Ratelimit-Remaining']) < 2:
            print("Rate limit reached")
        
            # Get the current UTC time with timezone info
            now_utc = datetime.datetime.now(datetime.timezone.utc)

            # Get the current time in seconds since the epoch
            utc_seconds = int(now_utc.timestamp())

            time_to_reset = int(response_headers['X-Ratelimit-Reset']) - utc_seconds

            print("Limit reached. Got to sleep!")

            time.sleep(time_to_reset+300)

            print("Wake up!")

        
        #if response_headers['X-Ratelimit-Remaining']:
            
        metadata = response.json()
        #print(metadata)
        
        # Extracting relevant metadata
        repo_info = {
            'name': metadata.get('name'),
            'description': metadata.get('description'),
            'language': metadata.get('language'),
            'stars': metadata.get('stargazers_count'),
            'url': metadata.get('html_url'),
            'topics': metadata.get('topics'),
            'github_url': f"https://github.com/{owner}/{repo}"
        }
        
        return repo_info
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from GitHub: {e}")
        return None
    
def get_repo_languages(owner, repo, token, info):
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    headers = {'Authorization': f'token {token}'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an error for HTTP error codes
        metadata_languages = response.json()
        info["languages"]=metadata_languages

        #print(metadata_languages)

        return info
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from GitHub: {e}")
        return None
    
def extract_metadata (github_url):

    pattern = r"https?://(www\.)?github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, github_url)
    
    if match:
        owner = match.group(2)
        repo = match.group(3)

        token = "Your token"
        
        metadata = get_repo_metadata(owner, repo, token)

        metadata = get_repo_languages(owner, repo, token, metadata)
        
    else:
        raise ValueError("Invalid GitHub URL format.")
    
    return metadata

if __name__ == "__main__":
    file_path = 'datasets/github_openaire.csv'

    init_position = 0
    end_position = 1
    batch_size = 10

    #Read until de first line indicates in init_position
     # Read the entire CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Return rows from index 9 to 99 (10th to 100th rows)
    selected_lines = df.iloc[init_position:end_position]

    if os.path.isfile('results/metadata_github_list.csv'):
        extended_df = pd.read_csv('results/metadata_github_list.csv')
    else:
        extended_df = pd.DataFrame()

    for url in selected_lines["codeRepositoryUrl"]:
        print("Analyzing repo:"+str(url))
        metadata_github_repo = extract_metadata(url)
        extended_df = extended_df._append(metadata_github_repo, ignore_index=True)
        extended_df.to_csv('metadata_github_list.csv',index=False)


    
    