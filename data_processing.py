# data_processing.py
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore')

from PIL import Image
import re
import cv2
import json
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing
import csv
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from chemprop.data import make_split_indices
from tdc.single_pred import ADME, Tox
from tqdm import tqdm
import glob          
import torch
#####################################################
random.seed(42)  # Very important for reproducibility
#####################################################
import requests
from io import StringIO
from pathlib import Path

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

current_path_beginning = os.getcwd().split("DEEPScreen")[0]
current_path_version = os.getcwd().split("DEEPScreen")[1].split(os.sep)[0]

project_file_path = os.path.join(current_path_beginning,"DEEPScreen"+current_path_version)
training_files_path = os.path.join(project_file_path,"training_files")
result_files_path = os.path.join(project_file_path,"result_files")
trained_models_path = os.path.join(project_file_path,"trained_models")


def get_chemblid_smiles_inchi_dict(smiles_inchi_fl):
    chemblid_smiles_inchi_dict = pd.read_csv(smiles_inchi_fl, sep=",", index_col=False).set_index('molecule_chembl_id').T.to_dict('list')
    return chemblid_smiles_inchi_dict


def save_comp_imgs_from_smiles(tar_id, comp_id, smiles, rotations, target_prediction_dataset_path, SIZE=300, rot_size=300):
    

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return
    
    base_path = os.path.join(target_prediction_dataset_path, tar_id, "imgs")
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    try:
        rotations_to_add = []
        for rot, suffix in rotations:
            if os.path.exists(os.path.join(base_path, f"{comp_id}{suffix}.png")): # Don't recreate images already done
                continue
            else:
                rotations_to_add.append((rot, suffix))
        if len(rotations_to_add) == 0:
            return
        image = Draw.MolToImage(mol, size=(SIZE, SIZE))
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        for rot, suffix in rotations_to_add:

            if rot != 0:
                full_image = np.full((rot_size, rot_size, 3), (255, 255, 255), dtype=np.uint8)
                gap = rot_size - SIZE
                (cX, cY) = (gap // 2, gap // 2)
                full_image[cY:cY + SIZE, cX:cX + SIZE] = image_bgr
                (cX, cY) = (rot_size // 2, rot_size // 2)
                M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
                full_image = cv2.warpAffine(full_image, M, (rot_size, rot_size), borderMode=cv2.INTER_LINEAR,
                                            borderValue=(255, 255, 255))
            else:
                full_image = image_bgr
            path_to_save = os.path.join(base_path, f"{comp_id}{suffix}.png")

            

            cv2.imwrite(path_to_save, full_image)
    except Exception as e:
        print(f"Error creating PNG for {comp_id}: {e}")

def initialize_dirs(targetid , target_prediction_dataset_path):
    if not os.path.exists(os.path.join(target_prediction_dataset_path, targetid, "imgs")):
        os.makedirs(os.path.join(target_prediction_dataset_path, targetid, "imgs"))

def process_smiles(smiles_data,augmentation_angle):

    current_smiles, compound_id, target_prediction_dataset_path, targetid,act_inact = smiles_data
    rotations = [(0, "_0"), *[(angle, f"_{angle}") for angle in range(augmentation_angle, 360, augmentation_angle)]]
    try:

        save_comp_imgs_from_smiles(targetid, compound_id, current_smiles, rotations, target_prediction_dataset_path)

        if not os.path.exists(os.path.join(target_prediction_dataset_path, targetid, "imgs","{}_0.png".format(compound_id))):

            print(f"{compound_id} image generation failed.")
            return compound_id
        
        else:
            return None
           
    except Exception as e:

        print(f"Error for {compound_id}: {e}")
        return compound_id

    
def generate_images(smiles_file, targetid, max_cores, target_prediction_dataset_path, augmentation_angle):
    df = pd.read_csv(smiles_file)
    smiles_data_list = [(row['canonical_smiles'], row['molecule_chembl_id'], target_prediction_dataset_path, targetid, row['act_inact_id']) for _, row in df.iterrows()]
    
    black_list = []
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = [executor.submit(process_smiles, s, augmentation_angle) for s in smiles_data_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Generating Images"):
            failed_id = f.result()
            if failed_id:
                black_list.append(failed_id)

    if black_list:
        df = df[~df['molecule_chembl_id'].isin(black_list)]
        df.to_csv(smiles_file, index=False)
    return df


def handle_group(g):
    if len(g) == 1:
        return g.iloc[0]
    elif len(g) == 2:
        return g.loc[g['pchembl_value'].idxmin()] 
    else:
        median_vals = g.median(numeric_only=True)
        row = g.iloc[0].copy()
        for col in median_vals.index:
            row[col] = median_vals[col]
        return row
    
def detect_deduplicate_and_save(input_csv):

    df = pd.read_csv(str(input_csv))
    df_cleaned = df.groupby(
        ['molecule_chembl_id', 'canonical_smiles'],
        group_keys=False
    ).apply(handle_group).reset_index(drop=True)

    base, ext = os.path.splitext(str(input_csv))
    output_csv = f"{base}_dup_detect{ext}"

    df_cleaned.to_csv(output_csv, index=False)
    print(f"Deduplicated file saved to {output_csv}")

    input_path = Path(input_csv)
    detect_csv = str(input_path.with_name(input_path.stem + "_dup_detect.csv"))
    print("New csv file name: ",output_csv)
                     
    return detect_csv

def apply_subsampling(act_list, inact_list, max_total_samples):
    """
    Apply subsampling to maintain 1:1 positive-negative ratio with max total samples.
    
    Args:
        act_list: List of active compound IDs
        inact_list: List of inactive compound IDs
        max_total_samples: Maximum total number of samples
        
    Returns:
        Tuple of (sampled_act_list, sampled_inact_list)
    """
    total_samples = len(act_list) + len(inact_list)
    
    if total_samples <= max_total_samples:
        print(f"Total samples ({total_samples}) <= max_total_samples ({max_total_samples}). No subsampling needed.")
        return act_list, inact_list
    
    samples_per_class = max_total_samples // 2
    
    sampled_act_list = random.sample(act_list, min(samples_per_class, len(act_list)))
    sampled_inact_list = random.sample(inact_list, min(samples_per_class, len(inact_list)))
    
    print(f"Subsampling applied: {len(act_list)} -> {len(sampled_act_list)} actives, {len(inact_list)} -> {len(sampled_inact_list)} inactives")
    
    return sampled_act_list, sampled_inact_list

def get_uniprot_chembl_sp_id_mapping(chembl_uni_prot_mapping_fl):
    id_mapping_fl = open(os.path.join(training_files_path, chembl_uni_prot_mapping_fl))
    lst_id_mapping_fl = id_mapping_fl.read().split("\n")
    id_mapping_fl.close()
    uniprot_to_chembl_dict = dict()
    for line in lst_id_mapping_fl[1:-1]:
        uniprot_id, chembl_id, prot_name, target_type = line.split("\t")
        if target_type=="SINGLE PROTEIN":
            if uniprot_id in uniprot_to_chembl_dict:
                uniprot_to_chembl_dict[uniprot_id].append(chembl_id)
            else:
                uniprot_to_chembl_dict[uniprot_id] = [chembl_id]
    return uniprot_to_chembl_dict

def get_chembl_uniprot_sp_id_mapping(chembl_mapping_fl):
    id_mapping_fl = open(os.path.join(training_files_path, chembl_mapping_fl))
    lst_id_mapping_fl = id_mapping_fl.read().split("\n")
    id_mapping_fl.close()
    chembl_to_uniprot_dict = dict()
    for line in lst_id_mapping_fl[1:-1]:
        uniprot_id, chembl_id, prot_name, target_type = line.split("\t")
        if target_type=="SINGLE PROTEIN":
            if chembl_id in chembl_to_uniprot_dict:
                chembl_to_uniprot_dict[chembl_id].append(uniprot_id)
            else:
                chembl_to_uniprot_dict[chembl_id] = [uniprot_id]
    return chembl_to_uniprot_dict

def get_act_inact_list_for_all_targets(fl):
    act_inact_dict = dict()
    with open(fl) as f:
        for line in f:
            if line.strip() != "":  
                parts = line.strip().split("\t")
                if len(parts) == 2:  
                    chembl_part, comps = parts
                    chembl_target_id, act_inact = chembl_part.split("_")
                    if act_inact == "act":
                        act_list = comps.split(",")
                        act_inact_dict[chembl_target_id] = [act_list, []]
                    else:
                        inact_list = comps.split(",")
                        act_inact_dict[chembl_target_id][1] = inact_list
    return act_inact_dict




def create_act_inact_files_for_targets(fl, target_id, pchembl_threshold, target_prediction_dataset_path):
    """
    Groups compounds into active and inactive lists based on pChEMBL threshold 
    and saves them in a format compatible with get_act_inact_list_for_all_targets.
    """
    # Create target directory
    target_dir = os.path.join(target_prediction_dataset_path, target_id)
    os.makedirs(target_dir, exist_ok=True)
    
    # Read the preprocessed activity data
    df = pd.read_csv(fl, sep=",", index_col=False)
    
    # Separate actives and inactives based on the threshold
    # Thresholding: Active >= threshold, Inactive < threshold
    act_rows = df[df['pchembl_value'] >= pchembl_threshold]['molecule_chembl_id'].unique().tolist()
    inact_rows = df[df['pchembl_value'] < pchembl_threshold]['molecule_chembl_id'].unique().tolist()
    
    # Define file path for the combined TSV
    combined_file_path = os.path.join(target_dir, f"{target_id}_preprocessed_filtered_act_inact_comps_pchembl_{pchembl_threshold}.tsv")
    
    # Write the file in the specific order your loader expects:
    # Line 1: TargetID_act \t comp1,comp2...
    # Line 2: TargetID_inact \t comp3,comp4...
    with open(combined_file_path, 'w') as f:
        f.write(f"{target_id}_act\t{','.join(act_rows)}\n")
        f.write(f"{target_id}_inact\t{','.join(inact_rows)}\n")
        
    # Also save a count file for quick manual inspection
    count_file_path = os.path.join(target_dir, f"{target_id}_preprocessed_filtered_act_inact_count_pchembl_{pchembl_threshold}.tsv")
    with open(count_file_path, 'w') as f:
        f.write(f"{target_id}\t{len(act_rows)}\t{len(inact_rows)}\n")
    
    print(f"--- Classification complete for {target_id} | Saved to: {combined_file_path} ---")
    return combined_file_path


def create_act_inact_files_similarity_based_neg_enrichment_threshold(act_inact_fl, blast_sim_fl, sim_threshold):

    data_point_threshold = 100
    uniprot_chemblid_dict = get_uniprot_chembl_sp_id_mapping("chembl27_uniprot_mapping.txt")
    chemblid_uniprot_dict = get_chembl_uniprot_sp_id_mapping("chembl27_uniprot_mapping.txt")
    all_act_inact_dict = get_act_inact_list_for_all_targets(act_inact_fl)
    new_all_act_inact_dict = dict()
    count = 0
    for targ in all_act_inact_dict.keys():
        act_list, inact_list = all_act_inact_dict[targ]
        if len(act_list)>=data_point_threshold and len(inact_list)>=data_point_threshold:
            count += 1
    

    seq_to_other_seqs_score_dict = dict()
    with open(os.path.join(training_files_path, blast_sim_fl)) as f:
        for line in f:
            parts = line.split("\t")
            u_id1, u_id2, score = parts[0].split("|")[1], parts[1].split("|")[1], float(parts[2])
            if u_id1!=u_id2:
                if u_id1 in seq_to_other_seqs_score_dict:
                    seq_to_other_seqs_score_dict[u_id1][u_id2] = score
                else:
                    seq_to_other_seqs_score_dict[u_id1] = dict()
                    seq_to_other_seqs_score_dict[u_id1][u_id2] = score
                if u_id2 in seq_to_other_seqs_score_dict:
                    seq_to_other_seqs_score_dict[u_id2][u_id1] = score
                else:
                    seq_to_other_seqs_score_dict[u_id2] = dict()
                    seq_to_other_seqs_score_dict[u_id2][u_id1] = score

    for u_id in seq_to_other_seqs_score_dict:
        seq_to_other_seqs_score_dict[u_id] = {k: v for k, v in sorted(seq_to_other_seqs_score_dict[u_id].items(), key=lambda item: item[1], reverse=True)}

    count = 0
    for chembl_target_id in all_act_inact_dict.keys():
        count += 1
        target_act_list, target_inact_list = all_act_inact_dict[chembl_target_id]
        target_act_list, target_inact_list = target_act_list[:], target_inact_list[:]
        uniprot_target_id = chemblid_uniprot_dict[chembl_target_id][0]
        if uniprot_target_id in seq_to_other_seqs_score_dict:
            for uniprot_other_target in seq_to_other_seqs_score_dict[uniprot_target_id]:
                if seq_to_other_seqs_score_dict[uniprot_target_id][uniprot_other_target]>=sim_threshold:
                    try:
                        other_target_id = uniprot_chemblid_dict[uniprot_other_target][0]
                        other_act_lst, other_inact_lst = all_act_inact_dict[other_target_id]
                        set_non_act_inact = set(other_inact_lst) - set(target_act_list)
                        set_new_inacts = set_non_act_inact - (set(target_inact_list) & set_non_act_inact)
                        target_inact_list.extend(list(set_new_inacts))
                    except:
                        pass
        new_all_act_inact_dict[chembl_target_id] = [target_act_list, target_inact_list]

    act_inact_comp_fl = open(os.path.join(training_files_path,act_inact_fl.split(".tsv")[0],"_blast_comp_",sim_threshold), "w")
    act_inact_count_fl = open(os.path.join(training_files_path,act_inact_fl.split(".tsv")[0],"_blast_count_",sim_threshold), "w")

    for targ in new_all_act_inact_dict.keys():
        if len(new_all_act_inact_dict[targ][0])>=data_point_threshold and len(new_all_act_inact_dict[targ][1])>=data_point_threshold:
            while "" in new_all_act_inact_dict[targ][0]:
                new_all_act_inact_dict[targ][0].remove("")

            while "" in new_all_act_inact_dict[targ][1]:
                new_all_act_inact_dict[targ][1].remove("")

            str_act = "{}_act\t".format(targ) + ",".join(new_all_act_inact_dict[targ][0])
            act_inact_comp_fl.write("{}\n".format(str_act))

            str_inact = "{}_inact\t".format(targ) + ",".join(new_all_act_inact_dict[targ][1])
            act_inact_comp_fl.write("{}\n".format(str_inact))

            str_act_inact_count = "{}\t{}\t{}\n".format(targ, len(new_all_act_inact_dict[targ][0]), len(new_all_act_inact_dict[targ][1]))
            act_inact_count_fl.write(str_act_inact_count)

    act_inact_count_fl.close()
    act_inact_comp_fl.close()

def get_uniprot_id_from_chembl(chembl_target_id: str) -> str:
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_target_id}.json"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    for comp in data.get("target_components", []):
        for xref in comp.get("target_component_xrefs", []):
            if xref.get("xref_src_db") == "UniProt":
                return xref.get("xref_id")
    raise ValueError(f"No UniProt ID found for {chembl_target_id}")

# 2) Fetch FASTA sequence from UniProt REST

def fetch_uniprot_sequence(uniprot_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    resp = requests.get(url)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    seq = ''.join(lines[1:])
    return seq

# 3) Submit sequence to EBI BLAST REST API and retrieve tabular results

def run_ebi_blast(email,seq: str, program: str = "blastp", database: str = "uniprotkb") -> pd.DataFrame:
    print("run_ebi_blast started")
    run_url = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/run"
    params = {
        'program': program,
        'database': database,
        'sequence': seq,
        'stype': 'protein',
        'email': email ,  # Your email adress
        'format': 'tsv'
    }
    r = requests.post(run_url, data=params)
    r.raise_for_status()
    job_id = r.text.strip()

    # Polling loop
    status_url = f"https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/status/{job_id}"
    while True:
        s = requests.get(status_url).text.strip()
        if s in ('FINISHED', 'ERROR'):
            break
        time.sleep(3)

    if s != 'FINISHED':
        raise RuntimeError(f"BLAST job {job_id} ended with status {s}")

    # Get the correct output format (TSV!)
    result_url = f"https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/result/{job_id}/tsv"
    out = requests.get(result_url)
    out.raise_for_status()

    df = pd.read_csv(StringIO(out.text), sep='\t')

    print("BLAST result columns:", df.columns)
    df = df.rename(columns={'Identities(%)': 'identity', 'Accession': 'hit_id'})

    return df


# 4) Filter hits >= threshold and extract UniProt IDs

def get_similar_uniprot_ids(df, threshold) -> list:
    print("get_similar_uniprot_ids started")
    print("df.columns:", df.columns)
    hits = df[(df['identity'] >= threshold) & (df['identity'] < 100)]
    # hit_id may contain full header, extract UniProt accession
    hits['uniprot_id'] = hits['hit_id'].apply(lambda x: x.split('|')[1] if '|' in x else x)
    
    hits_sorted = hits.sort_values(by='identity', ascending=False)
    
    similar_dict = dict(zip(hits_sorted['uniprot_id'], hits_sorted['identity']))

    print(len(similar_dict))
    
    return similar_dict

# 5) Map UniProt IDs to ChEMBL target IDs

def get_chembl_from_uniprot(uniprot_id: str) -> list:
    print("get_chembl_from_uniprot started")
    url = f"https://www.ebi.ac.uk/chembl/api/data/target.json?target_components.xrefs.xref_id={uniprot_id}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json().get('targets', [])
    return [t['target_id'] for t in data]

# 6) Fetch inactive compounds for a ChEMBL target

def fetch_inactive_compounds(chembl_target_id, pchembl_threshold) -> list:
    print("fetch_inactive_compounds started")
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        'target_id': chembl_target_id,
        'pchembl_value__lt': pchembl_threshold,
        'limit': 10000
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    results = r.json().get('activities', [])
    return [a['molecule_chembl_id'] for a in results]

def enrich_chemblid_smiles_dict(chemblid_smiles_dict, new_chembl_ids):
    
    valid_ids = []
    for chembl_id in new_chembl_ids:
        if chembl_id in chemblid_smiles_dict:
            valid_ids.append(chembl_id)
            continue
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
            r = requests.get(url)
            r.raise_for_status()
            mol_data = r.json()
            smiles = mol_data.get('molecule_structures', {}).get('canonical_smiles', None)
            if smiles:
                chemblid_smiles_dict[chembl_id] = ["dummy1", "dummy2", "dummy3", smiles]
                valid_ids.append(chembl_id)
            else:
                print(f"Warning: smiles not found for {chembl_id}")
        except Exception as e:
            print(f"Error fetching smiles for {chembl_id}: {e}")
    return chemblid_smiles_dict, valid_ids





def negative_enrichment_pipeline(chembl_target_id,
                                 percent_threshold,
                                 pchembl_threshold,
                                 original_actives,
                                 original_inactives,
                                 email,
                                 chemblid_smiles_dict):
    
    uni_id = get_uniprot_id_from_chembl(chembl_target_id)
    seq = fetch_uniprot_sequence(uni_id)
    df = run_ebi_blast(email, seq)
    similar_unis = get_similar_uniprot_ids(df, percent_threshold)
    sorted_unis = sorted(similar_unis.items(), key=lambda x: x[1], reverse=True)

    print("Number of similar proteins : ",len(sorted_unis))

    for uniprot_id, identity in similar_unis.items():
        print(f"UniProt ID: {uniprot_id}, Identity: {identity}")

    enriched_inactives = set()
    max_to_add = len(original_actives) - len(original_inactives)
    if max_to_add <= 0:
        return original_inactives, chemblid_smiles_dict

    for uniprot_id, identity in sorted_unis:

        print(f"UniProt ID: {uniprot_id}, Identity: {identity}%")
        chembl_targets = get_chembl_from_uniprot(uniprot_id)
        for ct in chembl_targets:
            new_inactives = fetch_inactive_compounds(ct, pchembl_threshold)
            for candidate in new_inactives:
                
                if len(enriched_inactives) >= max_to_add:
                    break
                if candidate in original_actives or candidate in original_inactives or candidate in enriched_inactives:
                    continue
                chemblid_smiles_dict, valid_ids = enrich_chemblid_smiles_dict(chemblid_smiles_dict, [candidate])
                if candidate in valid_ids:
                    enriched_inactives.add(candidate)
 
            if len(enriched_inactives) >= max_to_add:
                break
        if len(enriched_inactives) >= max_to_add:
            break

    combined_inactives = set(original_inactives) | enriched_inactives

    return list(combined_inactives), chemblid_smiles_dict


def create_final_randomized_training_val_test_sets(activity_data,max_cores,scaffold,targetid,target_prediction_dataset_path,dataset,no_fix_tdc ,pchembl_threshold,subsampling,max_total_samples,similarity_threshold,negative_enrichment,augmentation_angle,email):
    """
    split_dict : tdc dataset split object, dict of keys: string of training, valid, test; values: pd dataframes
    """
    if dataset == "moleculenet" or dataset =="tdc_adme" or dataset == "tdc_tox":

        if dataset == "moleculenet":
            pandas_df = pd.read_csv(activity_data)        
            pandas_df.rename(columns={pandas_df.columns[0]: "canonical_smiles", pandas_df.columns[-1]: "target"}, inplace=True)
            pandas_df = pandas_df[["canonical_smiles", "target"]]
        else:
            if dataset == "tdc_adme":
                data = ADME(name = targetid,path = os.path.join("training_files","target_training_datasets",targetid))
            if dataset == "tdc_tox":
                data = Tox(name = targetid,path = os.path.join("training_files","target_training_datasets",targetid))

            split = data.get_split(method = "scaffold" if scaffold else "random", frac = [0.8,0.1,0.1],seed = 42)
            split["train"]["split"] = "train"
            split["valid"]["split"] = "valid"
            split["test"]["split"] = "test" 
            pandas_df = pd.concat([split["train"],split["valid"],split["test"]])

            pandas_df.rename(columns={pandas_df.columns[1]: "canonical_smiles", pandas_df.columns[-2]: "target"}, inplace=True)
            pandas_df = pandas_df[["Drug_ID","canonical_smiles", "target","split"]].copy()

        pandas_df["molecule_chembl_id"] = [f"{targetid}{i+1}" for i in range(len(pandas_df))]

        act_ids = pandas_df[pandas_df["target"] == 1]["molecule_chembl_id"].tolist()
        inact_ids = pandas_df[pandas_df["target"] == 0]["molecule_chembl_id"].tolist()
        act_inact_dict = {targetid: [act_ids, inact_ids]}
        

        if dataset != "moleculenet": # Creating splits given by tdc
            tdc_split_dict = {}
            # This code sequence fixes tdc's mislabels.
            if no_fix_tdc: 
                for m in pandas_df.itertuples(index=False):
                    # Use canonical SMILES or Drug_ID as before
                    act_or_inact = "act" if m.target == 1 else "inact"
                    tdc_split_dict[m.molecule_chembl_id] = (m.split, act_or_inact)
            else:
                pandas_df = pandas_df.drop_duplicates(subset=["canonical_smiles","Drug_ID"], keep="first") # Drop duplicate rows that contain same smiles representation more than once
                for m in pandas_df.itertuples(index=False): 
                    m_drug = m.canonical_smiles
                    m_drug_id = m.Drug_ID  

                    in_a = (split["train"]["Drug_ID"] == m_drug_id).any()
                    in_b = (split["valid"]["Drug_ID"] == m_drug_id).any()
                    in_c = (split["test"]["Drug_ID"] == m_drug_id).any()

                    if in_a and not in_b and not in_c:
                        train_test_val_situation = "train"

                    elif in_b and not in_a and not in_c:
                        train_test_val_situation = "valid"

                    elif in_c and not in_a and not in_b:
                        train_test_val_situation = "test"

                    else:
                        in_a = (split["train"]["Drug"] == m_drug).any()
                        in_b = (split["valid"]["Drug"] == m_drug).any()
                        in_c = (split["test"]["Drug"] == m_drug).any()
                        if in_a:
                            train_test_val_situation = "train"

                        elif in_b:
                            train_test_val_situation = "valid"

                        elif in_c:
                            train_test_val_situation = "test"

                    act_or_inact = "act" if m.target == 1 else "inact"
                    tdc_split_dict[m.molecule_chembl_id] = (train_test_val_situation, act_or_inact)
        
            train_count = sum(1 for v in tdc_split_dict.values() if v[0] == "train")
            valid_count = sum(1 for v in tdc_split_dict.values() if v[0] == "valid")
            test_count  = sum(1 for v in tdc_split_dict.values() if v[0] == "test")

            orig_train_count = len(split["train"])
            orig_valid_count = len(split["valid"])
            orig_test_count  = len(split["test"])
            print("\n=== SPLIT CHECK ===")
            print(f"Original train size: {orig_train_count} | Assigned train size: {train_count}")
            print(f"Original valid size: {orig_valid_count} | Assigned valid size: {valid_count}")
            print(f"Original test size : {orig_test_count}  | Assigned test size : {test_count}")

            ok_train = (train_count == orig_train_count)
            ok_valid = (valid_count == orig_valid_count)
            ok_test  = (test_count == orig_test_count)

            if ok_train and ok_valid and ok_test:
                print("✔ Split control checked: Train/Valid/Test numbers match!")
            else:
                print("⚠ WARNING: Mismatch in split sizes! This is likely due to fixing of mislabels in TDC dataset.")

        moleculenet_dict = {}
        for i, row_ in pandas_df.iterrows():
            cid = row_["molecule_chembl_id"]
            smi = row_["canonical_smiles"]
            moleculenet_dict[cid] = ["dummy1", "dummy2", "dummy3", smi]
        chemblid_smiles_dict = moleculenet_dict
    else:
        df = pd.read_csv(activity_data)
    
        duplicates = df[df.duplicated(subset=['molecule_chembl_id','canonical_smiles'], keep=False)]
        
        if duplicates.empty:
            print("There is no duplicate rows in activity_data.csv.")

        else:

            print(f"Total number of duplicate rows: {len(duplicates)}")
            duplicate_ids = duplicates['molecule_chembl_id'].unique()
            print(duplicate_ids)

            activity_data = detect_deduplicate_and_save(activity_data)

            df = pd.read_csv(activity_data)
        
            duplicates = df[df.duplicated(subset=['molecule_chembl_id','canonical_smiles'], keep=False)]
        
            if duplicates.empty:
                print("Duplicate rows are handled")

        chemblid_smiles_dict = get_chemblid_smiles_inchi_dict(activity_data) 
    
        create_act_inact_files_for_targets(activity_data, targetid, pchembl_threshold, target_prediction_dataset_path) 

        act_inact_dict = get_act_inact_list_for_all_targets("{}/{}/{}_preprocessed_filtered_act_inact_comps_pchembl_{}.tsv".format(target_prediction_dataset_path, targetid, targetid, pchembl_threshold))

        print("Total act inact number : ",len(act_inact_dict))

    for tar in act_inact_dict:
        
        target_img_path = os.path.join(target_prediction_dataset_path, tar, "imgs")
        if not os.path.exists(target_img_path):
            os.makedirs(target_img_path)
        act_list, inact_list = act_inact_dict[tar]


        if (negative_enrichment and dataset !="moleculenet" and dataset !="tdc_adme" and dataset !="tdc_tox"):

            print("Before negative enrichment, length of the act and inact list")
            print("len act :" + str(len(act_list)))
            print("len inact :" + str(len(inact_list)))

            inact_list,chemblid_smiles_dict = negative_enrichment_pipeline(targetid,similarity_threshold,pchembl_threshold,act_list,inact_list,email,chemblid_smiles_dict)

            print("After negative enrichment, length of the act and inact list")
            print("len act :" + str(len(act_list)))
            print("len inact :" + str(len(inact_list)))


        # Apply subsampling if enabled
        
        if subsampling:
            act_list, inact_list = apply_subsampling(act_list, inact_list, max_total_samples)
            print(f"After subsampling - len act: {len(act_list)}, len inact: {len(inact_list)}")

        print("After subsampling, length of the act and inact list")
        print("len act :" + str(len(act_list)))
        print("len inact :" + str(len(inact_list)))

        if max_cores > multiprocessing.cpu_count():
            print(f"Warning: Maximum number of cores is {multiprocessing.cpu_count()}. Using maximum available cores.")
            max_cores = multiprocessing.cpu_count()

        directory = os.path.join(target_prediction_dataset_path,targetid)

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = "smilesfile.csv"
        smiles_file_csv = os.path.join(directory, file_name)

        smiles_file = smiles_file_csv

        with open(smiles_file_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["canonical_smiles", "molecule_chembl_id", "act_inact_id"])

            for root_id in act_list:
                try:
                    smi = chemblid_smiles_dict[root_id][3]
                    writer.writerow([smi, root_id, "1"])
                except KeyError:
                    continue

            for root_id in inact_list:
                try:
                    smi = chemblid_smiles_dict[root_id][3]
                    writer.writerow([smi, root_id, "0"])
                except KeyError:
                    continue
        
        initialize_dirs(targetid , target_prediction_dataset_path)

        generate_images(smiles_file, targetid, max_cores,target_prediction_dataset_path, augmentation_angle)

        training_act_comp_id_list = []
        val_act_comp_id_list = []
        test_act_comp_id_list = []
            
        training_inact_comp_id_list = []
        val_inact_comp_id_list = []
        test_inact_comp_id_list = []

        if not (dataset == "tdc_adme" or dataset == "tdc_tox"): # If the dataset is from TDC, then the splits are already provided and molecules should be put accordingly.
            
            (
                training_act_comp_id_list,
                val_act_comp_id_list,
                test_act_comp_id_list,
                training_inact_comp_id_list,
                val_inact_comp_id_list,
                test_inact_comp_id_list
            ) = train_val_test_split(smiles_file, scaffold, augmentation_angle)

        
            print("Train act len : ",len(training_act_comp_id_list))
            print("Train inact len : ",len(training_inact_comp_id_list))
            print("val act len : ",len(val_act_comp_id_list))
            print("val inact len : ",len(val_inact_comp_id_list))
            print("Test act len : ",len(test_act_comp_id_list))
            print("Test inact len : ",len(test_inact_comp_id_list))
            
        else:

            # Check for divisibility to avoid incomplete rotations
            if 360 % augmentation_angle != 0:
                raise ValueError(f"Error: 360 is not divisible by augmentation_angle ({augmentation_angle}).")
            
            cleaned_df = pd.read_csv(smiles_file)
            valid_molecule_ids = set(cleaned_df['molecule_chembl_id'].tolist())


            for molecule, (split, act_or_inact) in tdc_split_dict.items():
                
                # --- CRITICAL CHECK ---
                # Skip the molecule if it was removed during generate_images (failed image creation)
                if molecule not in valid_molecule_ids:
                    print(f"Skipping {molecule}: Image was not generated successfully.")
                    continue
                # ----------------------

                # Generate augmented IDs only for verified molecules
                augmented_ids = [f"{molecule}_{i}" for i in range(0, 360, augmentation_angle)]

                if split == "train" and act_or_inact == "act":
                    training_act_comp_id_list.extend(augmented_ids)
                elif split == "train" and act_or_inact == "inact":
                    training_inact_comp_id_list.extend(augmented_ids)
                elif split == "valid" and act_or_inact == "act":
                    val_act_comp_id_list.extend(augmented_ids)
                elif split == "valid" and act_or_inact == "inact":
                    val_inact_comp_id_list.extend(augmented_ids)
                elif split == "test" and act_or_inact == "act":
                    test_act_comp_id_list.extend(augmented_ids)
                elif split == "test" and act_or_inact == "inact":
                    test_inact_comp_id_list.extend(augmented_ids)
        
        print(tar, "all training act", len(act_list), len(training_act_comp_id_list), len(val_act_comp_id_list), len(test_act_comp_id_list))
        print(tar, "all training inact", len(inact_list), len(training_inact_comp_id_list), len(val_inact_comp_id_list), len(test_inact_comp_id_list))

        # JSON objesini hazırlıyoruz
        tar_train_val_test_dict = {
            "training": [],
            "validation": [],
            "test": []
        }

        tar_train_val_test_dict["training"].extend([[cid, 1] for cid in training_act_comp_id_list])
        tar_train_val_test_dict["training"].extend([[cid, 0] for cid in training_inact_comp_id_list])
       
        tar_train_val_test_dict["validation"].extend([[cid, 1] for cid in val_act_comp_id_list])
        tar_train_val_test_dict["validation"].extend([[cid, 0] for cid in val_inact_comp_id_list])
        
        tar_train_val_test_dict["test"].extend([[cid, 1] for cid in test_act_comp_id_list])
        tar_train_val_test_dict["test"].extend([[cid, 0] for cid in test_inact_comp_id_list])

    
        print(f"Final Counts for {tar}:")
        print(f"  Train: {len(tar_train_val_test_dict['training'])}")
        print(f"  Val:   {len(tar_train_val_test_dict['validation'])}")
        print(f"  Test:  {len(tar_train_val_test_dict['test'])}")

        dict_output_path = os.path.join(target_prediction_dataset_path, tar, 'train_val_test_dict.json')
        with open(dict_output_path, 'w') as fp:
            json.dump(tar_train_val_test_dict, fp)



def train_val_test_split(smiles_file, scaffold_split, augmentation_angle, split_ratios=(0.8, 0.1, 0.1)):
    """
    Splits data into train, validation, and test sets. 
    Ensures all rotations (augmented images) of a single molecule stay within the same set
    to prevent data leakage during model training.
    """
    # Defensive check: Ensure 360 is divisible by the angle
    if 360 % augmentation_angle != 0:
        raise ValueError(
            f"Invalid augmentation_angle: {augmentation_angle}. "
            f"360 must be perfectly divisible by the angle to ensure full rotation coverage."
        )

    # Load the cleaned dataset (only compounds with successful images)
    df = pd.read_csv(smiles_file)

    print("Splitting process started...")
    print(f"Total compounds in pool: {len(df)}")
    
    # Identify root molecule IDs for actives (1) and inactives (0)
    act_list = df[df['act_inact_id'] == 1]['molecule_chembl_id'].tolist()
    inact_list = df[df['act_inact_id'] == 0]['molecule_chembl_id'].tolist()

    print(f"Active root IDs: {len(act_list)}")
    print(f"Inactive root IDs: {len(inact_list)}")

    def expand_with_angles(id_list):
        """Generates augmented ID strings (e.g., CHEMBL123_90) from a list of root IDs."""
        augmented = []
        for root_id in id_list:
            for angle in range(0, 360, augmentation_angle):
                augmented.append(f"{root_id}_{angle}")
        return augmented

    if scaffold_split:
        print("--- Mode: Scaffold Balanced Split (Grouping by Molecule) ---")
        all_root_ids = act_list + inact_list
        # Retrieve SMILES for scaffold calculation from root IDs
        smiles_list = [df[df['molecule_chembl_id'] == cid]['canonical_smiles'].values[0] for cid in all_root_ids]
        labels = [1] * len(act_list) + [0] * len(inact_list)
        
        df_root = pd.DataFrame({'molecule_chembl_id': all_root_ids, 'smiles': smiles_list, 'label': labels})
        mols = [Chem.MolFromSmiles(s) for s in df_root['smiles']]
        
        # Chemprop v2.1+ returns a tuple of lists. 
        # Indices might be nested depending on the environment, so we flatten if necessary.
        indices = make_split_indices(mols, split="scaffold_balanced", sizes=split_ratios, seed=42)
        
        # Ensure indices are flat 1D lists to prevent 'Buffer wrong number of dimensions' error
        train_idx = indices[0][0] if isinstance(indices[0][0], (list, np.ndarray)) else indices[0]
        val_idx   = indices[1][0] if isinstance(indices[1][0], (list, np.ndarray)) else indices[1]
        test_idx  = indices[2][0] if isinstance(indices[2][0], (list, np.ndarray)) else indices[2]
        
        # Map indices to dataframes
        tr_df = df_root.iloc[train_idx]
        vl_df = df_root.iloc[val_idx]
        ts_df = df_root.iloc[test_idx]
        
        # Categorize root IDs by label within each split set
        tr_act_roots = tr_df[tr_df['label'] == 1]['molecule_chembl_id'].tolist()
        vl_act_roots = vl_df[vl_df['label'] == 1]['molecule_chembl_id'].tolist()
        ts_act_roots = ts_df[ts_df['label'] == 1]['molecule_chembl_id'].tolist()
        
        tr_inact_roots = tr_df[tr_df['label'] == 0]['molecule_chembl_id'].tolist()
        vl_inact_roots = vl_df[vl_df['label'] == 0]['molecule_chembl_id'].tolist()
        ts_inact_roots = ts_df[ts_df['label'] == 0]['molecule_chembl_id'].tolist()

    else:
        print("--- Mode: Full Random Split (Grouping by Molecule) ---")
        # Shuffle root IDs so all rotations move together
        random.shuffle(act_list)
        random.shuffle(inact_list)
        
        def split_roots(lst):
            n = len(lst)
            t = int(n * split_ratios[0])
            v = int(n * split_ratios[1])
            # Basic proportional slicing
            return lst[:t], lst[t:t+v], lst[t+v:]

        # Perform split at the root molecule level
        tr_act_roots, vl_act_roots, ts_act_roots = split_roots(act_list)
        tr_inact_roots, vl_inact_roots, ts_inact_roots = split_roots(inact_list)

    # FINAL STEP: Apply data augmentation to root IDs after the split is finalized.
    # All rotations of a single molecule are guaranteed to exist in ONLY one set.
    tr_act = expand_with_angles(tr_act_roots)
    vl_act = expand_with_angles(vl_act_roots)
    ts_act = expand_with_angles(ts_act_roots)
    
    tr_inact = expand_with_angles(tr_inact_roots)
    vl_inact = expand_with_angles(vl_inact_roots)
    ts_inact = expand_with_angles(ts_inact_roots)

    print("-" * 35)
    print(f"Augmented Train   (Act/Inact): {len(tr_act)} / {len(tr_inact)}")
    print(f"Augmented Val     (Act/Inact): {len(vl_act)} / {len(vl_inact)}")
    print(f"Augmented Test    (Act/Inact): {len(ts_act)} / {len(ts_inact)}")
    print("-" * 35)

    return tr_act, vl_act, ts_act, tr_inact, vl_inact, ts_inact




class DEEPScreenDataset(Dataset):
    def __init__(self, target_id, train_val_test,parent_path = os.path.join(training_files_path,"target_training_datasets")):
        self.target_id = target_id
        self.train_val_test = train_val_test
        self.dataset_path = os.path.join(parent_path,target_id)
        self.train_val_test_folds = json.load(open(os.path.join(self.dataset_path, "train_val_test_dict.json")))

        if train_val_test == "all":
            self.compid_list = [compid_label[0] for compid_label in self.train_val_test_folds]
            self.label_list = [compid_label[1] for compid_label in self.train_val_test_folds]
        else:
            self.compid_list = [compid_label[0] for compid_label in self.train_val_test_folds[train_val_test]]
            self.label_list = [compid_label[1] for compid_label in self.train_val_test_folds[train_val_test]]
    
    def __len__(self):
        return len(self.compid_list)

    def __getitem__(self, index):
        comp_id = self.compid_list[index]
        
        img_paths = [os.path.join(self.dataset_path, "imgs", "{}.png".format(comp_id))]        

        img_path = random.choice([path for path in img_paths if os.path.exists(path)])      
            
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found for compound ID: {comp_id}")
        img_arr = np.array(Image.open(img_path))
        
        if img_arr is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {img_path}")

        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        label = self.label_list[index]

        return img_arr, label, comp_id

def get_train_test_val_data_loaders(target_id, batch_size=32):
    training_dataset = DEEPScreenDataset(target_id, "training")
    validation_dataset = DEEPScreenDataset(target_id, "validation")
    test_dataset = DEEPScreenDataset(target_id, "test")

    train_sampler = SubsetRandomSampler(range(len(training_dataset)))
    train_loader = DataLoader(training_dataset, batch_size=batch_size, sampler=train_sampler,generator=g,worker_init_fn=seed_worker)
    
    validation_sampler = SubsetRandomSampler(range(len(validation_dataset)))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler,generator=g,worker_init_fn=seed_worker)

    test_sampler = SubsetRandomSampler(range(len(test_dataset)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,generator=g,worker_init_fn=seed_worker)

    return train_loader, validation_loader, test_loader

def get_training_target_list(chembl_version):
    target_df = pd.read_csv(os.path.join(training_files_path, "{}_training_target_list.txt".format(chembl_version)), index_col=False, header=None)
    
    return list(target_df[0])


class PredictionDataset(Dataset):
    """
    Dataset specifically for Prediction/Inference.
    1. Loads ALL rotations for compounds in the specified split.
    2. Uses a dictionary for label lookup.
    3. Filters images based on which compounds are in label_dict.
    """
    def __init__(self, target_id, parent_path, label_dict=None):
        self.target_id = target_id
        self.dataset_path = os.path.join(parent_path, target_id)
        self.img_dir = os.path.join(self.dataset_path, "imgs")
        self.label_dict = label_dict  # Expects { 'CompID': 1, ... }

        # Find ALL images (e.g., CHEMBL123_0.png, CHEMBL123_10.png)
        all_image_paths = glob.glob(os.path.join(self.img_dir, "*.png"))
        
        # Filter images based on label_dict (only include compounds in the split)
        if self.label_dict is not None:
            # Extract compound IDs that should be included (from label_dict keys)
            valid_compound_ids = set(self.label_dict.keys())
            
            # Filter image paths to only include those matching valid compound IDs
            self.all_image_paths = []
            for img_path in all_image_paths:
                filename = os.path.basename(img_path)
                comp_id = filename.rsplit('.')[0]  # "CHEMBL123_10.png" -> "CHEMBL123_10"
                
                # Remove rotation suffix to get base compound ID
                base_comp_id = re.sub(r'_\d+$', '', comp_id)  # "CHEMBL123_10" -> "CHEMBL123"
                
                # Only include if this compound is in the label_dict
                if base_comp_id in valid_compound_ids:
                    self.all_image_paths.append(img_path)
            
            print(f"Filtered to {len(self.all_image_paths)} images from {len(all_image_paths)} total images")
            print(f"Covering {len(valid_compound_ids)} unique compounds in the specified split")
        else:
            # No filtering - use all images
            self.all_image_paths = all_image_paths
            print(f"No split filtering - using all {len(self.all_image_paths)} images")
        
        if len(self.all_image_paths) == 0:
            print(f"Warning: No images found in {self.img_dir} for the specified split")

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        img_path = self.all_image_paths[index]
        
        # Parse CompID from filename: "CHEMBL123_10.png" -> "CHEMBL123_10"
        filename = os.path.basename(img_path)
        comp_id = filename.rsplit('.')[0]
        
        # Image Loading & Normalization
        try:
            img = Image.open(img_path)
            img_arr = np.array(img)
            img_arr = img_arr / 255.0
            img_arr = img_arr.transpose((2, 0, 1)) # (C, H, W)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            img_arr = np.zeros((3, 200, 200)) # Dummy

        # Label Lookup
        label = -1 # Default (Unknown)
        base_comp_id = re.sub(r'_\d+$', '', comp_id)
        if self.label_dict and base_comp_id in self.label_dict.keys():
            label = self.label_dict[base_comp_id]
            
        # Return torch tensor immediately
        return torch.tensor(img_arr).float(), label, comp_id

def load_deepscreen_labels(target_id, parent_path,split):
    """
    Parses 'train_val_test_dict.json' located in the target folder.
    Structure: {"training": [["ID", 1]...], "validation": ...}
    Returns flattened dict: {'ID': 1}
    """
    json_path = os.path.join(parent_path, target_id, "train_val_test_dict.json")

    if not os.path.exists(json_path):
        return None
    
    print(f"Loading labels from {json_path}...")
    try:
        data = json.load(open(json_path))
        label_dict = {}
        
        # Flatten the dictionary
        for split_key, sample_list in data.items():
            if split_key!=split:
                continue
            for item in sample_list:
                if len(item) >= 2:
                    comp_id = item[0]
                    label = int(item[1])
                    comp_id = re.sub(r'_\d+$', '', comp_id)
                    label_dict[comp_id] = label
        return label_dict

    except Exception as e:
        print(f"Error parsing label file: {e}")
        return None

def get_prediction_loader(target_id, parent_path, label_dict=None, batch_size=32):
    """
    Returns a DataLoader using the PredictionDataset.
    
    Args:
        target_id: Target identifier
        parent_path: Path to parent directory
        label_dict: Dictionary mapping compound IDs to labels.
                   If provided, only images for compounds in this dict will be loaded.
                   This enables split-specific prediction.
        batch_size: Batch size for DataLoader
    
    Returns:
        DataLoader or None if no images found
    """
    dataset = PredictionDataset(target_id, parent_path, label_dict)
    if len(dataset) == 0:
        return None
    # Shuffle=False is important for reproducibility in prediction
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
