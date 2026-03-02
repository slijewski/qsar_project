import logging
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from rdkit.Chem import Draw

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def lipinski_one(mol):
    desc_MolWt = Descriptors.MolWt(mol)
    desc_MolLogP = Descriptors.MolLogP(mol)
    desc_NumHDonors = Lipinski.NumHDonors(mol)
    desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
    return [desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors]

def calculate_fp_one(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((0,))
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('outputs/egfr_model.pkl')
        selection = joblib.load('outputs/variance_selection.pkl')
        return model, selection
    except:
        return None, None

st.set_page_config(page_title="EGFR Activity Predictor", layout="wide")

st.title("💊 EGFR Bioactivity Predictor (QSAR)")
st.markdown("Predicts molecule activity against EGFR based on ChEMBL data.")

st.sidebar.header("Input Data")
input_smiles = st.sidebar.text_area("Enter SMILES code", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")

if st.sidebar.button("Predict Activity"):
    model, selection = load_model()

    if model is None:
        st.error("Model not found! Run `03_model_building.py` first to train the model.")
    else:
        try:
            mol = Chem.MolFromSmiles(input_smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(400, 200))
                st.image(img, caption="Molecule Structure")

                desc = lipinski_one(mol)
                fp = calculate_fp_one(mol)

                fp_transformed = selection.transform(fp)

                prediction = model.predict(fp_transformed)
                pIC50 = prediction[0]

                ic50_nm = 10**(9 - pIC50)

                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted pIC50", value=f"{pIC50:.2f}")
                with col2:
                    st.metric(label="Estimated IC50", value=f"{ic50_nm:.2f} nM")

                if pIC50 > 6.0:
                    st.success("✅ **ACTIVE Compound** (Potential Lead)")
                else:
                    st.warning("❌ **INACTIVE Compound**")

                st.markdown("### Physicochemical Properties (Lipinski's Rule of 5)")
                st.table(pd.DataFrame([desc], columns=["MW", "LogP", "NumHDonors", "NumHAcceptors"]))

            else:
                st.error("Invalid SMILES code!")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Enter SMILES code in the sidebar and click 'Predict Activity'.")

st.markdown("""
---
*Author: Sebastian Lijewski, PhD*
""")
