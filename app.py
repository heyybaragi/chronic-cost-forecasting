import streamlit as st
import pandas as pd
import sys

sys.path.append('.')
from src.features import build_patient_features, CHRONIC_COLS
from src.predict import predict_cost_and_tier

st.set_page_config(page_title="Chronic Disease Cost Forecasting", page_icon="🏥", layout="wide")

st.title("Chronic Disease Cost Forecasting")
st.write("Predict a Medicare beneficiary's 12-month cost trajectory and risk tier, built on CMS synthetic claims data.")
st.markdown(
    "This tool predicts a Medicare patient's healthcare cost for the coming year, "
    "based on their chronic conditions and prior spending. It then sorts them into a "
    "Low, Medium, or High risk tier, the kind of prioritization real care management "
    "teams use to decide who needs outreach first. Built on CMS's public synthetic "
    "Medicare claims data (SynPUF), using patients' 2008 and 2009 history to predict 2010 cost."
)
st.markdown("---")

@st.cache_data
def load_population_data():
    return pd.read_csv('data/processed/test_predictions.csv')


population_df = load_population_data()

# ---------------- SIDEBAR: PATIENT INPUT FORM ----------------

st.sidebar.header("Patient Profile")

sample_patients = {
    "Custom (enter manually)": None,
    "Low-risk example": {
        'age': 68, 'sex': 1, 'race': 1, 'esrd': False,
        'chronic_conditions': {c: False for c in CHRONIC_COLS},
        'cost_2008': 400.0, 'cost_2009': 350.0
    },
    "High-cost multimorbid example": {
        'age': 79, 'sex': 2, 'race': 1, 'esrd': False,
        'chronic_conditions': {c: True for c in CHRONIC_COLS[:6]} | {c: False for c in CHRONIC_COLS[6:]},
        'cost_2008': 9800.0, 'cost_2009': 12400.0
    }
}

selected_sample = st.sidebar.selectbox("Quick example, or enter manually below", list(sample_patients.keys()))
preset = sample_patients[selected_sample]

age = st.sidebar.slider("Age", 25, 100, value=preset['age'] if preset else 72)
sex = st.sidebar.radio("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female",
                        index=0 if not preset else preset['sex'] - 1)
race = st.sidebar.selectbox("Race code", options=[1, 2, 3, 4, 5],
                             index=0 if not preset else preset['race'] - 1)
esrd = st.sidebar.checkbox("End-Stage Renal Disease (ESRD)", value=preset['esrd'] if preset else False)

st.sidebar.subheader("Chronic Conditions")
chronic_labels = {
    'SP_ALZHDMTA': "Alzheimer's / Dementia", 'SP_CHF': "Heart Failure",
    'SP_CHRNKIDN': "Chronic Kidney Disease", 'SP_CNCR': "Cancer",
    'SP_COPD': "COPD", 'SP_DEPRESSN': "Depression", 'SP_DIABETES': "Diabetes",
    'SP_ISCHMCHT': "Ischemic Heart Disease", 'SP_OSTEOPRS': "Osteoporosis",
    'SP_RA_OA': "Rheumatoid / Osteoarthritis", 'SP_STRKETIA': "Stroke / TIA"
}

chronic_conditions = {}
for col in CHRONIC_COLS:
    default = preset['chronic_conditions'][col] if preset else False
    chronic_conditions[col] = st.sidebar.checkbox(chronic_labels[col], value=default)

st.sidebar.subheader("Prior Year Cost")
cost_2008 = st.sidebar.number_input("2008 total cost ($)", min_value=0.0,
                                     value=preset['cost_2008'] if preset else 2000.0, step=100.0)
cost_2009 = st.sidebar.number_input("2009 total cost ($)", min_value=0.0,
                                     value=preset['cost_2009'] if preset else 2000.0, step=100.0)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("Predict Cost & Risk Tier", type="primary")

# ---------------- MAIN AREA: TABS ----------------

tab1, tab2 = st.tabs(["Predict for a Patient", "Population Overview"])

with tab1:
    if predict_clicked:
        patient_input = {
            'age': age, 'sex': sex, 'race': race, 'esrd': esrd,
            'chronic_conditions': chronic_conditions,
            'cost_2008': cost_2008, 'cost_2009': cost_2009
        }

        patient_features = build_patient_features(patient_input)
        result = predict_cost_and_tier(patient_features)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted 2010 Cost", f"${result['predicted_cost']:,.2f}")
        with col2:
            tier_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.metric("Risk Tier", f"{tier_colors[result['risk_tier']]} {result['risk_tier']}")

        st.write(f"This patient's predicted 12-month cost places them in the **{result['risk_tier']}** risk tier.")
    else:
        st.info("Fill in the patient profile in the sidebar, then click **Predict Cost & Risk Tier** to see results.")

with tab2:
    st.subheader("How the model performs across the population")

    tier_counts = population_df['risk_tier'].value_counts().reindex(['Low', 'Medium', 'High'])
    tier_avg_cost = population_df.groupby('risk_tier')['actual_cost'].mean().reindex(['Low', 'Medium', 'High'])

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Patients per risk tier**")
        st.bar_chart(tier_counts)
    with col2:
        st.write("**Average actual cost by tier**")
        st.bar_chart(tier_avg_cost)

    total_cost = population_df['actual_cost'].sum()
    high_tier_cost = population_df[population_df['risk_tier'] == 'High']['actual_cost'].sum()
    high_tier_pct = len(population_df[population_df['risk_tier'] == 'High']) / len(population_df) * 100
    high_cost_pct = high_tier_cost / total_cost * 100

    st.write(
        f"In this test set, the **High** risk tier makes up about **{high_tier_pct:.0f}% of patients** "
        f"but accounts for **{high_cost_pct:.0f}% of total cost**. This is the group care management "
        f"teams would prioritize first, a small group with an outsized share of spending."
    )