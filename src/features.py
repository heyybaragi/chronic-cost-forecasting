import pandas as pd

FEATURE_COLUMNS = [
    'age', 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_ESRD_IND',
    'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD',
    'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS',
    'SP_RA_OA', 'SP_STRKETIA', 'chronic_count',
    'cost_2008', 'cost_2009', 'cost_trend'
]

CHRONIC_COLS = [
    'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD',
    'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS',
    'SP_RA_OA', 'SP_STRKETIA'
]


def build_patient_features(patient_input: dict) -> pd.DataFrame:
    """
    Takes a dict of raw patient inputs (from the Streamlit form or anywhere else)
    and returns a single-row DataFrame shaped exactly like the model's training data.

    Expected keys in patient_input:
        age (int)
        sex (int, 1 or 2)
        race (int, 1-5)
        esrd (bool)
        chronic_conditions (dict of the 11 SP_ flags -> bool)
        cost_2008 (float)
        cost_2009 (float)
    """
    row = {
        'age': patient_input['age'],
        'BENE_SEX_IDENT_CD': patient_input['sex'],
        'BENE_RACE_CD': patient_input['race'],
        'BENE_ESRD_IND': int(patient_input['esrd']),
    }

    for col in CHRONIC_COLS:
        row[col] = int(patient_input['chronic_conditions'][col])

    row['chronic_count'] = sum(row[col] for col in CHRONIC_COLS)
    row['cost_2008'] = patient_input['cost_2008']
    row['cost_2009'] = patient_input['cost_2009']
    row['cost_trend'] = row['cost_2009'] - row['cost_2008']

    df = pd.DataFrame([row])
    return df[FEATURE_COLUMNS]