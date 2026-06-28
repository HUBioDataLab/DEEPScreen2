import os
import glob
import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client

def get_smiles_from_chembl(chembl_id):
    """ChEMBL ID'den SMILES çeker."""
    try:
        molecule = new_client.molecule
        res = molecule.filter(molecule_chembl_id=chembl_id).only(['molecule_structures'])
        if res:
            return res[0]['molecule_structures']['canonical_smiles']
    except Exception:
        return None
    return None

def get_smiles_from_pubchem(name):
    """DILI veya diğer ID'leri PubChem üzerinden aramayı dener."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=5)
        if response.status_status == 200:
            return response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception:
        return None
    return None

def update_csv_with_smiles(data_dir, output_csv="molecules.csv"):
    print(f"🔍 {data_dir} taranıyor...")
    search_pattern = os.path.join(data_dir, "**", "*_0.npz")
    files = glob.glob(search_pattern, recursive=True)
    
    mol_ids = sorted(list(set([os.path.basename(f).replace("_0.npz", "") for f in files])))
    print(f"Bulunan benzersiz molekül sayısı: {len(mol_ids)}")

    results = []
    for m_id in mol_ids:
        print(f"📡 Veri çekiliyor: {m_id}...", end=" ", flush=True)
        
        # 1. Yol: ChEMBL API
        smiles = get_smiles_from_chembl(m_id)
        
        # 2. Yol: PubChem (ChEMBL bulamazsa veya DILI id'si ise)
        if not smiles:
            smiles = get_smiles_from_pubchem(m_id)
            
        if smiles:
            print("✅ Bulundu")
        else:
            print("❌ Bulunamadı")
            
        results.append({"molecule_id": m_id, "smiles": smiles})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n📁 İşlem bitti! '{output_csv}' kontrol edebilirsin.")

if __name__ == "__main__":
    update_csv_with_smiles("./shap_numpy_data")