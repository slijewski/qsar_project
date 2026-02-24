import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import AllChem

def lipinski(smiles, verbose=False):
    moldata = []
    valid_indices = []
    
    for i, elem in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(elem)
            if mol is not None:
                # Sanitize mol to check validity
                Chem.SanitizeMol(mol)
                moldata.append(mol)
                valid_indices.append(i)
        except:
            pass
            
    if len(moldata) == 0:
        return pd.DataFrame(), []
       
    baseData = np.arange(1,1)
    i = 0  
    for mol in moldata:        
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt, 
                        desc_MolLogP, 
                        desc_NumHDonors, 
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i=i+1      
    
    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    
    return descriptors, valid_indices

def calculate_fingerprints(smiles_list):
    # Morgan Fingerprints (radius=2, nBits=1024)
    fps = []
    valid_indices = []
    for i, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((0,))
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_indices.append(i)
            
    X = np.array(fps)
    return X, valid_indices

if __name__ == "__main__":
    print("Loading data...")
    try:
        df = pd.read_csv("data/egfr_bioactivity_data.csv")
    except FileNotFoundError:
        print("Error: data/egfr_bioactivity_data.csv not found. Run 01_data_collection.py first.")
        exit(1)
    
    # 1. Calculate Lipinski descriptors
    print("Calculating Lipinski descriptors...")
    df_lipinski, valid_idx_lipinski = lipinski(df.canonical_smiles)
    
    if df_lipinski.empty:
        print("Error: No valid molecules found in dataset.")
        exit(1)
        
    # Filter original dataframe to keep only valid molecules
    df = df.iloc[valid_idx_lipinski].reset_index(drop=True)
    df_combined = pd.concat([df, df_lipinski], axis=1)
    
    # Categorize activity
    # Active: pChEMBL > 6 (IC50 < 1000 nM)
    
    # Check if we have pchembl_value, if not, calculate from standard_value
    if 'pchembl_value' not in df.columns or df['pchembl_value'].isnull().all():
        # Ensure standard_units is nM
        df_combined = df_combined[df_combined.standard_units == 'nM']
        df_combined['pIC50'] = 9 - np.log10(df_combined['standard_value'])
    else:
        df_combined['pIC50'] = df_combined['pchembl_value']

    # Bioactivity Class
    def categorize(val):
        if val >= 6.0:
            return 'active'
        else:
            return 'inactive'
            
    df_combined['class'] = df_combined['pIC50'].apply(categorize)
    
    print("Class distribution:")
    print(df_combined['class'].value_counts())
    
    # Save data with descriptors
    df_combined.to_csv("data/egfr_data_lipinski.csv", index=False)
    
    # 2. Generate fingerprints for ML
    print("Generating Morgan Fingerprints...")
    X, valid_idx = calculate_fingerprints(df_combined.canonical_smiles)
    
    # Save fingerprints as DataFrame
    df_fp = pd.DataFrame(X, columns=[f'FP_{i}' for i in range(1024)])
    
    # Keep only valid ones
    df_final = df_combined.iloc[valid_idx].reset_index(drop=True)
    df_fp['pIC50'] = df_final['pIC50']
    
    df_fp.to_csv('data/egfr_fingerprints.csv', index=False)
    print("Done. Data saved to data/egfr_fingerprints.csv")
