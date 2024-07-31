import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import random
import cv2
import numpy as np

# Dosya yollarını belirtin
current_path_beginning = os.getcwd().split("DEEPScreen")[0]
current_path_version = os.getcwd().split("DEEPScreen")[1].split("/")[0]

project_file_path = "{}DEEPScreen{}".format(current_path_beginning, current_path_version)
training_files_path = "{}/training_files".format(project_file_path)
result_files_path = "{}/result_files".format(project_file_path)
trained_models_path = "{}/trained_models".format(project_file_path)

activity_csv_path = "{}/target_training_datasets/CHEMBL286/activity_data.csv".format(training_files_path)

def get_chemblid_smiles_inchi_dict(smiles_inchi_fl):
    chemblid_smiles_inchi_dict = pd.read_csv(smiles_inchi_fl, sep=",", index_col=False).set_index('molecule_chembl_id').T.to_dict('list')
    return chemblid_smiles_inchi_dict

def create_act_inact_files_for_all_targets(df, act_threshold=10.0, inact_threshold=20.0):
    print("")
    print("Creating act and inact files for all targets")
    print("")

    act_rows_df = df[df["value"] <= act_threshold]
    inact_rows_df = df[df["value"] >= inact_threshold]
    target_act_inact_comp_dict = dict()

    for ind, row in act_rows_df.iterrows():
        chembl_tid = row['target_chembl_id']
        chembl_cid = row['molecule_chembl_id']

        if chembl_tid in target_act_inact_comp_dict:
            target_act_inact_comp_dict[chembl_tid][0].append(chembl_cid)
        else:
            target_act_inact_comp_dict[chembl_tid] = [[], []]
            target_act_inact_comp_dict[chembl_tid][0].append(chembl_cid)

    for ind, row in inact_rows_df.iterrows():
        chembl_tid = row['target_chembl_id']
        chembl_cid = row['molecule_chembl_id']
        if chembl_tid in target_act_inact_comp_dict:
            target_act_inact_comp_dict[chembl_tid][1].append(chembl_cid)
        else:
            target_act_inact_comp_dict[chembl_tid] = [[], []]
            target_act_inact_comp_dict[chembl_tid][1].append(chembl_cid)

    return target_act_inact_comp_dict

def get_act_inact_list_for_all_targets(act_inact_dict):
    return act_inact_dict

def create_final_randomized_training_val_test_sets(activity_csv_path, smiles_inchi_fl, act_threshold=10.0, inact_threshold=20.0):
    print("")
    print("Creating final randomized training, validation and test sets")
    print("")

    df = pd.read_csv(activity_csv_path)
    chemblid_smiles_dict = df.set_index('molecule_chembl_id')['canonical_smiles'].to_dict()
    act_inact_dict = create_act_inact_files_for_all_targets(df, act_threshold, inact_threshold)
    act_inact_dict = get_act_inact_list_for_all_targets(act_inact_dict)

    for tar in act_inact_dict:
        tar_path = os.path.join(training_files_path, "target_training_datasets", tar)
        os.makedirs(os.path.join(tar_path, "imgs"), exist_ok=True)
        act_list, inact_list = act_inact_dict[tar]
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
        rotations = [(0, "0"), *[(angle, f"{angle}") for angle in range(10, 360, 10)]]
        for comp_id in training_act_comp_id_list:
            try:
                save_comp_imgs_from_smiles(tar, comp_id, chemblid_smiles_dict[comp_id], rotations, training_files_path)
                tar_train_val_test_dict["training"].append([comp_id, 1])
            except Exception as e:
                print(f"Error creating training image for {comp_id}: {e}")
        for comp_id in val_act_comp_id_list:
            try:
                save_comp_imgs_from_smiles(tar, comp_id, chemblid_smiles_dict[comp_id], rotations, training_files_path)
                tar_train_val_test_dict["validation"].append([comp_id, 1])
            except Exception as e:
                print(f"Error creating validation image for {comp_id}: {e}")
        for comp_id in test_act_comp_id_list:
            try:
                save_comp_imgs_from_smiles(tar, comp_id, chemblid_smiles_dict[comp_id], rotations, training_files_path)
                tar_train_val_test_dict["test"].append([comp_id, 1])
            except Exception as e:
                print(f"Error creating test image for {comp_id}: {e}")
        for comp_id in training_inact_comp_id_list:
            try:
                save_comp_imgs_from_smiles(tar, comp_id, chemblid_smiles_dict[comp_id], rotations, training_files_path)
                tar_train_val_test_dict["training"].append([comp_id, 0])
            except Exception as e:
                print(f"Error creating training image for {comp_id}: {e}")
        for comp_id in val_inact_comp_id_list:
            try:
                save_comp_imgs_from_smiles(tar, comp_id, chemblid_smiles_dict[comp_id], rotations, training_files_path)
                tar_train_val_test_dict["validation"].append([comp_id, 0])
            except Exception as e:
                print(f"Error creating validation image for {comp_id}: {e}")
        for comp_id in test_inact_comp_id_list:
            try:
                save_comp_imgs_from_smiles(tar, comp_id, chemblid_smiles_dict[comp_id], rotations, training_files_path)
                tar_train_val_test_dict["test"].append([comp_id, 0])
            except Exception as e:
                print(f"Error creating test image for {comp_id}: {e}")
        random.shuffle(tar_train_val_test_dict["training"])
        random.shuffle(tar_train_val_test_dict["validation"])
        random.shuffle(tar_train_val_test_dict["test"])

        with open(os.path.join(tar_path, 'train_val_test_dict.json'), 'w') as fp:
            json.dump(tar_train_val_test_dict, fp)
        print(f"train_val_test_dict.json created at {os.path.join(tar_path, 'train_val_test_dict.json')}")

def save_comp_imgs_from_smiles(tar_id, comp_id, smiles, rotations, target_prediction_dataset_path, SIZE=300):
    print(f"Generating image for compound {comp_id} with SMILES {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return

    Draw.DrawingOptions.atomLabelFontSize = 55
    Draw.DrawingOptions.dotsPerAngstrom = 100
    Draw.DrawingOptions.bondLineWidth = 1.5

    base_path = os.path.join(target_prediction_dataset_path, "target_training_datasets", tar_id, "imgs")

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    try:
        image = Draw.MolToImage(mol, size=(SIZE, SIZE))
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        for rot, suffix in rotations:
            if rot != 0:
                (cX, cY) = (SIZE // 2, SIZE // 2)
                M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
                rotated_image = cv2.warpAffine(image_bgr, M, (SIZE, SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            else:
                rotated_image = image_bgr

            path_to_save = os.path.join(base_path, f"{comp_id}{suffix}.png")
            cv2.imwrite(path_to_save, rotated_image)
            print(f"Image saved at {path_to_save}")
    except Exception as e:
        print(f"Error creating PNG for {comp_id}: {e}")

if __name__ == "__main__":
    create_final_randomized_training_val_test_sets(activity_csv_path, activity_csv_path)
