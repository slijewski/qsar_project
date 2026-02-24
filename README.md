# 🧪 EGFR Bioactivity Prediction (QSAR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

## 📖 Introduction

This project implements a **QSAR (Quantitative Structure-Activity Relationship)** approach to predict the potency of chemical molecules against the **EGFR (Epidermal Growth Factor Receptor)**. Overexpression of EGFR is a key driver in many cancer types (including lung cancer), making the design of new inhibitors for this protein a cornerstone of modern drug discovery.

## ⚖️ Justification

Traditional drug discovery methods are extremely costly and time-consuming. Applying Machine Learning (ML) allows for early-stage virtual screening of thousands of molecules "in silico," significantly accelerating the drug discovery process and reducing laboratory research costs.

## 🛠️ Methods and Technologies

The project combines knowledge from Cheminformatics and Data Science:

- **Data Collection**: Automated retrieval of bioactivity data (IC50) from the **ChEMBL** database using a dedicated API.
- **Cheminformatics (RDKit)**:
  - Calculation of physicochemical descriptors (Lipinski's Rule of 5).
  - Generation of **Morgan Fingerprints** (1024-bit vectors) representing the 2D structure of molecules.
- **Machine Learning (Scikit-Learn)**:
  - Utilizing the **Random Forest Regressor** algorithm to predict pIC50 values.
  - Data optimization by removing features with low variance.
- **Deployment**: An interactive **Streamlit** dashboard for predicting drug potency based on SMILES codes.

## 🚀 Installation & Usage

### Installation

1. Navigate to the project folder:

   ```bash
   cd qsar_project
   ```

2. Install the required libraries:

   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

### Usage

1. Run the application:

   ```bash
   streamlit run app.py
   ```

2. Examples:
   - Aspirin: CC(=O)Oc1ccccc1C(=O)O - doesn't possess EGFR inhibitory activity
   - Paracetamol: CC(=O)Nc1ccc(O)cc1 - doesn't possess EGFR inhibitory activity
   - Gefitinib: O=C(c1ccc(Nc2cncc3ccccc23)cc1)Nc1ccc(Cl)c(F)c1 - possesses EGFR inhibitory activity
   - Erlotinib: COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC - known EGFR inhibitor
   - Afatinib: CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4 - possesses EGFR inhibitory activity
   - CSCCOC1=C(OCCSC)C=C2C(=C1)N=CN=C2N(C)C1=CC=CC(C=C)=C1 - a molecule not described by the science, similar to Erlotinib, predicted as active by the model.

## 📈 Conclusions

- The **Random Forest** model effectively identifies key structural fragments of molecules responsible for EGFR inhibition.
- Converting IC50 to a logarithmic scale (pIC50) improved the numerical stability of the model.
- The application serves as a practical tool supporting medicinal chemists in rapid verification of new research hypotheses.

## 📁 Repository Structure

```text
├── .python-version     # Python version pin (uv)
├── 01_data_collection.py # ChEMBL API data extraction
├── 02_eda_descriptors.py # Lipinski descriptors calculation
├── 03_model_building.py # Random Forest Regressor training
├── app.py              # Streamlit prediction dashboard
├── requirements.txt    # Project dependencies
├── uv.lock             # Lockfile for reproducible environment
├── data/               # Bioactivity datasets
└── outputs/            # Serialized models and scalers
```

## 📜 Acknowledgments

Data derived from ChEMBL. This project is intended for **educational and research purposes only** and should not be used as a substitute for professional medical advice or clinical decision-making.

---

## 👨‍🔬Author

Sebastian Lijewski
PhD in Pharmaceutical Sciences
