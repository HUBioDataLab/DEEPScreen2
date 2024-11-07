import requests
import pandas as pd
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

"""
The input file containing ChEMBL IDs should be a plain text (.txt) file.
The file should be formatted as follows:
- Each line should contain exactly one ChEMBL ID.
- No extra spaces or empty lines should be included.
- The file should have a .txt extension.

Example of a valid input file:
-------------------------------
CHEMBL25
CHEMBL192
CHEMBL1234567
CHEMBL345
-------------------------------

Note: Other file formats such as .csv, .xlsx, or .json are not supported.
Ensure that the file is saved as .txt for proper processing.

Example commands to run the script:
-----------------------------------

1. Using a .txt file with ChEMBL IDs:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --assay_type=B --pchembl_threshold=6.0 --output_file=activity_data.csv

2. Specifying multiple ChEMBL IDs directly in the command:
   python chembl_downloading.py --target_chembl_id=CHEMBL25,CHEMBL192 --assay_type=B --pchembl_threshold=6.0 --output_file=activity_data.csv

3. Combining a .txt file with additional specified ChEMBL IDs:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --target_chembl_id=CHEMBL345 --assay_type=B --pchembl_threshold=6.0 --output_file=activity_data.csv

4. Specifying a custom output file name:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --output_file=custom_output.csv

5. Limiting the number of CPU cores used:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --max_cores=4 --output_file=activity_data.csv

6. Changing the assay type filter:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --assay_type=A --output_file=activity_data.csv
"""

def fetch_activities(target_chembl_ids, assay_types, pchembl_threshold):
    print("Starting to fetch activities from ChEMBL...")  # Process start message
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        'target_chembl_id__in': ','.join(target_chembl_ids),
        'assay_type__in': ','.join(assay_types),
        'pchembl_value__isnull': 'false',
        'only': 'molecule_chembl_id,pchembl_value,target_chembl_id,bao_label'
    }
    
    activities = []
    while True:
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
            break
        
        data = response.json()
        if 'activities' in data:
            activities.extend(data['activities'])
           
        else:
            print("No activities found.")
            break
        
        if 'page_meta' in data and data['page_meta']['next']:
            params['offset'] = data['page_meta']['offset'] + data['page_meta']['limit']
        else:
            break

    if activities:
        df = pd.DataFrame(activities)
        if 'pchembl_value' in df.columns:
            df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
            df = df[df['pchembl_value'].notnull() & (df['pchembl_value'] >= pchembl_threshold)]
            df.drop(columns=['bao_label'], errors='ignore', inplace=True)
        else:
            print("pchembl_value column not found.")
            return pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    print("Finished fetching activities.")  # Process completion message
    return df

def fetch_smiles(compound_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{compound_id}.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        smiles = data.get('molecule_structures', {}).get('canonical_smiles', None)
        return compound_id, smiles
    else:
        print(f"Failed to fetch data for {compound_id}. HTTP Status Code: {response.status_code}")
        return compound_id, None

def check_and_download_smiles(compound_ids):
    print("Starting to download SMILES...")  # SMILES download start message
    smiles_data = []

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(fetch_smiles, compound_ids))
        
    for compound_id, smiles in results:
        if smiles:
            smiles_data.append((compound_id, smiles))
    
    print("Finished downloading SMILES.")  # SMILES download completion message
    return smiles_data

def read_chembl_ids_from_file(file_path):
    if os.path.exists(file_path):
        print(f"Reading ChEMBL IDs from {file_path}...")  # File read message
        with open(file_path, 'r') as file:
            chembl_ids = [line.strip() for line in file.readlines() if line.strip()]
            return chembl_ids
    else:
        print(f"File {file_path} does not exist.")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ChEMBL activity data and SMILES")
    parser.add_argument('--target_chembl_id', type=str, help="Target ChEMBL ID(s) to search for, comma-separated")
    parser.add_argument('--assay_type', type=str, default='B', help="Assay type(s) to search for, comma-separated")
    parser.add_argument('--pchembl_threshold', type=float, default=6.0, help="Threshold for pChembl value to determine active/inactive")
    parser.add_argument('--output_file', type=str, default='activity_data.csv', help="Output file to save activity data")
    parser.add_argument('--max_cores', type=int, default=multiprocessing.cpu_count() - 1, help="Maximum number of CPU cores to use")
    parser.add_argument('--smiles_input_file', type=str, help="Path to txt file containing ChEMBL IDs")

    args = parser.parse_args()

    # Define base directory based on the location of the script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    target_chembl_ids = []
    
    if args.target_chembl_id:
        target_chembl_ids.extend(args.target_chembl_id.split(','))
    
    if args.smiles_input_file:
        file_chembl_ids = read_chembl_ids_from_file(args.smiles_input_file)
        target_chembl_ids.extend(file_chembl_ids)
    
    assay_types = args.assay_type.split(',')

    for chembl_id in target_chembl_ids:
        output_dir = os.path.join(base_dir, 'training_files', 'target_training_datasets', chembl_id)
        output_path = os.path.join(output_dir, args.output_file)
        
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping download.")
            continue

        data = fetch_activities([chembl_id], assay_types, args.pchembl_threshold)
        
        if not data.empty:
            compound_ids = data['molecule_chembl_id'].unique().tolist()
            smiles_data = check_and_download_smiles(compound_ids)
            
            if smiles_data:
                # Only create directory if there is data to save
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                smiles_df = pd.DataFrame(smiles_data, columns=["molecule_chembl_id", "canonical_smiles"])
                data = data.merge(smiles_df, on='molecule_chembl_id')
                
                data.to_csv(output_path, index=False)
                print(f"Activity data for {chembl_id} saved to {output_path}")
            else:
                print(f"No SMILES data found for {chembl_id}.")
        else:
            print(f"No activity data found for {chembl_id}.")
