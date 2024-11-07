# data_processing.py

import os
import sys
import cv2
import json
import torch
import random
import warnings
import subprocess
import numpy as np
import pandas as pd
from operator import itemgetter
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, SequentialSampler
from rdkit import Chem
from rdkit.Chem import Draw
from concurrent.futures import ProcessPoolExecutor
import time
import argparse
import multiprocessing
import csv
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import chemprop

warnings.filterwarnings(action='ignore')
current_path_beginning = os.getcwd().split("DEEPScreen")[0]
current_path_version = os.getcwd().split("DEEPScreen")[1].split("/")[0]

project_file_path = "{}DEEPScreen{}".format(current_path_beginning, current_path_version)
training_files_path = "{}/training_files".format(project_file_path)
result_files_path = "{}/result_files".format(project_file_path)
trained_models_path = "{}/trained_models".format(project_file_path)

IMG_SIZE = 300


def get_chemblid_smiles_inchi_dict(smiles_inchi_fl):
    chemblid_smiles_inchi_dict = pd.read_csv(smiles_inchi_fl, sep=",", index_col=False).set_index('molecule_chembl_id').T.to_dict('list')
    return chemblid_smiles_inchi_dict


def save_comp_imgs_from_smiles(tar_id, comp_id, smiles, rotations, target_prediction_dataset_path, SIZE=300, rot_size=300):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return
    Draw.DrawingOptions.atomLabelFontSize = 55
    Draw.DrawingOptions.dotsPerAngstrom = 100
    Draw.DrawingOptions.bondLineWidth = 1.5
   
    base_path = os.path.join(target_prediction_dataset_path, tar_id, "imgs")
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    try:
        image = Draw.MolToImage(mol, size=(SIZE, SIZE))
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        for rot, suffix in rotations:
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
            #print(path_to_save)
            cv2.imwrite(path_to_save, full_image)
    except Exception as e:
        print(f"Error creating PNG for {comp_id}: {e}")

def initialize_dirs(targetid , target_prediction_dataset_path):
    if not os.path.exists(os.path.join(target_prediction_dataset_path, targetid, "imgs")):
        os.makedirs(os.path.join(target_prediction_dataset_path, targetid, "imgs"))

    f = open(os.path.join(target_prediction_dataset_path, targetid, "prediction_dict.json"), "w+")
    json_dict = {"prediction": list()}
    json_object = json.dumps(json_dict)
    f.write(json_object)
    f.close()
def process_smiles(smiles_data):
    current_smiles, compound_id, target_prediction_dataset_path, targetid,act_inact,test_val_train_situation = smiles_data
    rotations = [(0, "_0"), *[(angle, f"_{angle}") for angle in range(10, 360, 10)]]
    local_dict = {test_val_train_situation: []}
    try:
            
        save_comp_imgs_from_smiles(targetid, compound_id, current_smiles, rotations, target_prediction_dataset_path)

        for i in range(0,360,10):
            local_dict[test_val_train_situation].append([compound_id + "_" + str(i), int(act_inact)])

    except:
        
        pass
    return local_dict
    
def generate_images(smiles_file, targetid, max_cores,tar_train_val_test_dict,target_prediction_dataset_path):

    
    smiles_list = pd.read_csv(smiles_file)["canonical_smiles"].tolist()
    compound_ids = pd.read_csv(smiles_file)["molecule_chembl_id"].tolist()
    act_inact_situations = pd.read_csv(smiles_file)["act_inact_id"].tolist()
    test_val_train_situations = pd.read_csv(smiles_file)["test_val_train"].tolist()
    smiles_data_list = [(smiles, compound_ids[i], target_prediction_dataset_path, targetid,act_inact_situations[i],test_val_train_situations[i]) for i, smiles in enumerate(smiles_list)]

    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        results = list(executor.map(process_smiles, smiles_data_list))
    end_time = time.time()

    for result in results:
        for key, value in result.items():
            tar_train_val_test_dict[key].extend(value)

    print(f"Time taken for all: {end_time - start_time}")
    total_image_count = len(smiles_list) * len([(0, ""), *[(angle, f"_{angle}") for angle in range(10, 360, 10)]])
    print(f"Total images generated: {total_image_count}")


def create_preprocessed_bioact_file(chembl_filtered_bioact_fl, chembl_version):
    raw_dataset_df = pd.read_csv(chembl_filtered_bioact_fl, sep=",", index_col=False)
    annot_dict = dict()
    for ind, row in raw_dataset_df.iterrows():
        chembl_tid = row["target_chembl_id"]
        chembl_cid = row["molecule_chembl_id"]
        standard_units = row["value"]
        row["year"] = row["year"]

        if standard_units in ["uM", "nM", "M"]:
            if standard_units == "nM":
                row["value"] = round(row["value"] / pow(10,3), 3)
            elif standard_units == "M":
                row["value"] = round(row["value"] / pow(10,6), 3)
            else:
                row["value"] = round(row["value"], 3)
            row["standard_units"] = "uM"
            try:
                annot_dict["{},{}".format(chembl_tid, chembl_cid)].append(list(row))
            except:
                annot_dict["{},{}".format(chembl_tid, chembl_cid)] = [list(row)]

    out_fl = open("{}/{}_preprocessed_filtered_bioactivity_dataset.tsv".format(training_files_path, chembl_version), "w")
    out_fl.write("\t".join(list(raw_dataset_df.columns)))
    for key in annot_dict.keys():
        if len(annot_dict[key])>1:
            median_std_val = 0.0

            annot_dict[key]  = sorted(annot_dict[key], key=itemgetter(6))

            if len(annot_dict[key])%2==1:
                median = int(len(annot_dict[key])/2)
                median_bioactivity = annot_dict[key][median]
                out_fl.write("\n" + "\t".join([str(col) for col in median_bioactivity]))

            else:
                median = int(len(annot_dict[key])/2)

                median_std_val = (annot_dict[key][median][6]+annot_dict[key][median-1][6])/2
                annot_dict[key][median][6] = median_std_val
                median_bioactivity = annot_dict[key][median]
                out_fl.write("\n" + "\t".join([str(col) for col in median_bioactivity]))
        else:
            out_fl.write("\n"+"\t".join([str(col) for col in annot_dict[key][0]]))

    out_fl.close()

def get_uniprot_chembl_sp_id_mapping(chembl_uni_prot_mapping_fl):
    id_mapping_fl = open("{}/{}".format(training_files_path, chembl_uni_prot_mapping_fl))
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
    id_mapping_fl = open("{}/{}".format(training_files_path, chembl_mapping_fl))
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

def get_act_inact_list_for_a_target(target, fl):
    act_list = []
    inact_list = []

    with open("{}/{}".format(training_files_path, fl))  as f:
        for line in f:
            if line != "":
                line=line.split("\n")[0]
                chembl_part, comps = line.split("\t")
                chembl_target_id, act_inact = chembl_part.split("_")
                if chembl_target_id == target:
                    if act_inact == "act":
                        act_list = comps.split(",")
                    else:
                        inact_list = comps.split(",")
                        break

    return act_list, inact_list

def get_act_inact_list_for_all_targets(fl):
    act_inact_dict = dict()
    with open(fl) as f:
        for line in f:
            if line.strip() != "":  # Satırın boş olmadığından emin ol
                parts = line.strip().split("\t")
                if len(parts) == 2:  # Satırın doğru şekilde ayrıştırıldığından emin ol
                    chembl_part, comps = parts
                    chembl_target_id, act_inact = chembl_part.split("_")
                    if act_inact == "act":
                        act_list = comps.split(",")
                        act_inact_dict[chembl_target_id] = [act_list, []]
                    else:
                        inact_list = comps.split(",")
                        act_inact_dict[chembl_target_id][1] = inact_list
    return act_inact_dict


def create_act_inact_files_for_targets(fl, target_id, chembl_version, pchembl_threshold=6, target_prediction_dataset_path=None):
    pre_filt_chembl_df = pd.read_csv(fl, sep=",", index_col=False)
    act_rows_df = pre_filt_chembl_df[pre_filt_chembl_df["pchembl_value"] >= pchembl_threshold]
    inact_rows_df = pre_filt_chembl_df[pre_filt_chembl_df["pchembl_value"] < pchembl_threshold]
    target_act_inact_comp_dict = dict()

    for ind, row in act_rows_df.iterrows():
        chembl_tid = row['target_chembl_id']
        chembl_cid = row['molecule_chembl_id']

        if chembl_tid in target_act_inact_comp_dict:
            target_act_inact_comp_dict[chembl_tid][0].add(chembl_cid)
        else:
            target_act_inact_comp_dict[chembl_tid] = [set(), set()]
            target_act_inact_comp_dict[chembl_tid][0].add(chembl_cid)

    for ind, row in inact_rows_df.iterrows():
        chembl_tid = row['target_chembl_id']
        chembl_cid = row['molecule_chembl_id']
        if chembl_tid in target_act_inact_comp_dict:
            target_act_inact_comp_dict[chembl_tid][1].add(chembl_cid)
        else:
            target_act_inact_comp_dict[chembl_tid] = [set(), set()]
            target_act_inact_comp_dict[chembl_tid][1].add(chembl_cid)

    act_inact_comp_fl = open("{}/{}/{}_preprocessed_filtered_act_inact_comps_pchembl_{}.tsv".format(target_prediction_dataset_path,target_id, chembl_version,  pchembl_threshold), "w")
    act_inact_count_fl = open("{}/{}/{}_preprocessed_filtered_act_inact_count_pchembl_{}.tsv".format(target_prediction_dataset_path, target_id, chembl_version, pchembl_threshold), "w")

    for targ in target_act_inact_comp_dict:
        str_act = "{}_act\t".format(targ) + ",".join(target_act_inact_comp_dict[targ][0])
        act_inact_comp_fl.write("{}\n".format(str_act))

        str_inact = "{}_inact\t".format(targ) + ",".join(target_act_inact_comp_dict[targ][1])
        act_inact_comp_fl.write("{}\n".format(str_inact))

        str_act_inact_count = "{}\t{}\t{}\n".format(targ, len(target_act_inact_comp_dict[targ][0]), len(target_act_inact_comp_dict[targ][1]))
        act_inact_count_fl.write(str_act_inact_count)

    act_inact_count_fl.close()
    act_inact_comp_fl.close()


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
    with open("{}/{}".format(training_files_path, blast_sim_fl)) as f:
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
                        other_target_chembl_id = uniprot_chemblid_dict[uniprot_other_target][0]
                        other_act_lst, other_inact_lst = all_act_inact_dict[other_target_chembl_id]
                        set_non_act_inact = set(other_inact_lst) - set(target_act_list)
                        set_new_inacts = set_non_act_inact - (set(target_inact_list) & set_non_act_inact)
                        target_inact_list.extend(list(set_new_inacts))
                    except:
                        pass
        new_all_act_inact_dict[chembl_target_id] = [target_act_list, target_inact_list]

    act_inact_comp_fl = open("{}/{}_blast_comp_{}.txt".format(training_files_path, act_inact_fl.split(".tsv")[0], sim_threshold), "w")
    act_inact_count_fl = open("{}/{}_blast_count_{}.txt".format(training_files_path, act_inact_fl.split(".tsv")[0], sim_threshold), "w")

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

def create_final_randomized_training_val_test_sets(activity_data,max_cores,targetid,target_prediction_dataset_path, pchembl_threshold=6):

    chemblid_smiles_dict = get_chemblid_smiles_inchi_dict(activity_data) 
    
    create_act_inact_files_for_targets(activity_data, targetid, "chembl", pchembl_threshold, target_prediction_dataset_path) 

    act_inact_dict = get_act_inact_list_for_all_targets("{}/{}/{}_preprocessed_filtered_act_inact_comps_pchembl_{}.tsv".format(target_prediction_dataset_path, targetid, "chembl", pchembl_threshold))
    
    for tar in act_inact_dict:
        
        target_img_path = os.path.join(target_prediction_dataset_path, tar, "imgs")
        if not os.path.exists(target_img_path):
            os.makedirs(target_img_path)
        act_list, inact_list = act_inact_dict[tar]
        print("len act :" + str(len(act_list)))
        print("len inact :" + str(len(inact_list)))
        if len(inact_list) >= len(act_list):
            inact_list = inact_list[:len(act_list)]
        else:
            act_list = act_list[:int(len(inact_list) * 1.5)]

        random.shuffle(act_list)
        random.shuffle(inact_list)


        act_training_validation_size = int(0.8 * len(act_list))
        act_training_size = int(0.8 * act_training_validation_size)
        act_val_size = act_training_validation_size - act_training_size
        training_act_comp_id_list = act_list[:act_training_size]
        val_act_comp_id_list = act_list[act_training_size:act_training_size+act_val_size]
        test_act_comp_id_list = act_list[act_training_size+act_val_size:]

        inact_training_validation_size = int(0.8 * len(inact_list))
        inact_training_size = int(0.8 * inact_training_validation_size)
        inact_val_size = inact_training_validation_size - inact_training_size
        training_inact_comp_id_list = inact_list[:inact_training_size]
        val_inact_comp_id_list = inact_list[inact_training_size:inact_training_size+inact_val_size]
        test_inact_comp_id_list = inact_list[inact_training_size+inact_val_size:]

        print(tar, "all training act", len(act_list), len(training_act_comp_id_list), len(val_act_comp_id_list), len(test_act_comp_id_list))
        print(tar, "all training inact", len(inact_list), len(training_inact_comp_id_list), len(val_inact_comp_id_list), len(test_inact_comp_id_list))
        tar_train_val_test_dict = dict()
        tar_train_val_test_dict["training"] = []
        tar_train_val_test_dict["validation"] = []
        tar_train_val_test_dict["test"] = []
        

        parser = argparse.ArgumentParser()

    # Usage format:
    # --dataset_file <dataset_file_path> --max_cores <max_cores> --target_prediction_dataset_path <target_prediction_dataset_path> --protein_name <protein_name>


        directory = "{}/{}".format(target_prediction_dataset_path,targetid)

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = "smilesfile.csv"
        last_smiles_file = os.path.join(directory, file_name)
        print(last_smiles_file)
        with open(last_smiles_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["canonical_smiles", "molecule_chembl_id", "act_inact_id","test_val_train"])

        
        for comp_id in training_act_comp_id_list:
            with open(last_smiles_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([chemblid_smiles_dict[comp_id][3], comp_id, "1","training"])
            
        for comp_id in val_act_comp_id_list:
            with open(last_smiles_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([chemblid_smiles_dict[comp_id][3], comp_id, "1","validation"])
            
        for comp_id in test_act_comp_id_list:
            with open(last_smiles_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([chemblid_smiles_dict[comp_id][3], comp_id, "1","test"])
            
        for comp_id in training_inact_comp_id_list:
            with open(last_smiles_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([chemblid_smiles_dict[comp_id][3], comp_id, "0","training"])
            
        for comp_id in val_inact_comp_id_list:
            with open(last_smiles_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([chemblid_smiles_dict[comp_id][3], comp_id, "0","validation"])
            
        for comp_id in test_inact_comp_id_list:
            with open(last_smiles_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([chemblid_smiles_dict[comp_id][3], comp_id, "0","test"])

        #parser.add_argument('--dataset_file', type=str, default="{}/smilesfile.csv".format(directory), help='Path to the dataset file')
        #parser.add_argument('--max_cores', type=int, default = max_cores, help='Maximum number of cores to use')

        #args = parser.parse_args()

        if max_cores > multiprocessing.cpu_count():
            print(f"Warning: Maximum number of cores is {multiprocessing.cpu_count()}. Using maximum available cores.")
            max_cores = multiprocessing.cpu_count()

        smiles_file = last_smiles_file
        
        initialize_dirs(targetid , target_prediction_dataset_path)

        

        generate_images(smiles_file , targetid , max_cores , tar_train_val_test_dict,target_prediction_dataset_path)
        print(tar_train_val_test_dict)
        random.shuffle(tar_train_val_test_dict["training"])
        random.shuffle(tar_train_val_test_dict["validation"])
        random.shuffle(tar_train_val_test_dict["test"])

        with open(os.path.join(target_prediction_dataset_path, tar, 'train_val_test_dict.json'), 'w') as fp:
            json.dump(tar_train_val_test_dict, fp)
       
def train_val_test_split(smiles_file, target_column_number=1, scaffold_split=False):
    """
    Split data into train/val/test sets using either random or scaffold-based splitting
    
    Args:
        smiles_file: Path to CSV file containing SMILES and target data
        target_column_number: Column index for target values (default=1) 
        scaffold_split: Whether to use scaffold-based splitting (default=False)
    
    Returns:
        Lists of compound IDs for train/val/test splits
    """
    df = pd.read_csv(smiles_file)
    df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle with fixed seed
    
    # Get SMILES and compound IDs
    smiles = df["canonical_smiles"].tolist()
    compound_ids = df["molecule_chembl_id"].tolist()
    
    if scaffold_split:
        # Create MoleculeDatapoints for scaffold splitting
        molecule_list = []
        for i, smile in enumerate(smiles):
            molecule_list.append(chemprop.data.data.MoleculeDatapoint(
                smiles=[smile],
                targets=[1], # Dummy target since we just need structure
                id=compound_ids[i]
            ))
        molecule_dataset = chemprop.data.data.MoleculeDataset(molecule_list)
        
        # Perform scaffold split
        train, val, test = chemprop.data.scaffold.scaffold_split(
            data=molecule_dataset,
            sizes=(0.8, 0.1, 0.1),
            seed=42,
            balanced=True
        )
        
        # Extract compound IDs from split datasets
        train_ids = [mol.id[0] for mol in train]
        val_ids = [mol.id[0] for mol in val] 
        test_ids = [mol.id[0] for mol in test]
        
    else:
        # Random split
        n = len(compound_ids)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        train_ids = compound_ids[:train_size]
        val_ids = compound_ids[train_size:train_size + val_size]
        test_ids = compound_ids[train_size + val_size:]

    return train_ids, val_ids, test_ids


class DEEPScreenDataset(Dataset):
    def __init__(self, target_id, train_val_test):
        self.target_id = target_id
        self.train_val_test = train_val_test
        self.training_dataset_path = "{}/target_training_datasets/{}".format(training_files_path, target_id)
        self.train_val_test_folds = json.load(open(os.path.join(self.training_dataset_path, "train_val_test_dict.json")))
        self.compid_list = [compid_label[0] for compid_label in self.train_val_test_folds[train_val_test]]
        self.label_list = [compid_label[1] for compid_label in self.train_val_test_folds[train_val_test]]

    def __len__(self):
        return len(self.compid_list)

    def __getitem__(self, index):
        comp_id = self.compid_list[index]
        
        # Tüm açılar için görüntüleri okuyun
        #img_paths = [os.path.join(self.training_dataset_path, "imgs", "{}_{}.png".format(comp_id, angle)) for angle in range(0, 360, 10)]
        img_paths = [os.path.join(self.training_dataset_path, "imgs", "{}.png".format(comp_id))]

        
        img_path = random.choice([path for path in img_paths if os.path.exists(path)])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found for compound ID: {comp_id}")

        img_arr = cv2.imread(img_path)
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
    train_loader = DataLoader(training_dataset, batch_size=batch_size, sampler=train_sampler)
    
    validation_sampler = SubsetRandomSampler(range(len(validation_dataset)))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

    test_sampler = SubsetRandomSampler(range(len(test_dataset)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, validation_loader, test_loader

def get_training_target_list(chembl_version):
    target_df = pd.read_csv(os.path.join(training_files_path, "{}_training_target_list.txt".format(chembl_version)), index_col=False, header=None)
    # print(target_df)
    # print(list(target_df[0]), len(list(target_df[0])))
    return list(target_df[0])
