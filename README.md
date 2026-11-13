markdown
# Audit-First ML for Pb/Cd Biochar Adsorption

Official implementation of the paper "Audit-First Machine Learning for Pb and Cd Adsorption Capacity Prediction on Biochar."

## Data source

Mendeley Data, DOI: 10.17632/hp36d23mgm.1
Download the ZIP, extract, and place `Supplementary Material.xlsx` in `data/`.

## Reproduce

```bash
pip install -r requirements.txt
python 00_inspect.py
python 01_audit.py
python 02_preprocess.py
python 03_split.py
python 04_train_compare.py
python 05_shap_analysis.py
python 06_pdp.py
python 07_screening.py
Key finding
The Pb sheet stores Qe as an Excel formula. Raw reading leaves 91% missing. The pipeline reconstructs Qe = (C0-Ce)*V/m, recovering 349 records. Grouped CV then shows no model achieves positive test R² on Pb. This is the paper's point: audit before modelling.
Files
•	config.py — paths and constants
•	00–07 — pipeline scripts
•	results/tables/ — CSV outputs
•	results/figures/ — PNG outputs
•	results/models/ — saved .joblib models
