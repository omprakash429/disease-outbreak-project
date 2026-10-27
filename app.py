"""
app.py  -  Streamlit web interface.

Run:  streamlit run app.py
(You must run  python train_model.py  once first.)
"""

import json
import os
import joblib
import pandas as pd
import streamlit as st

from src.predict import predict_one

MODEL_PATH = os.path.join("models", "outbreak_model.joblib")
METRICS_PATH = os.path.join("reports", "metrics.json")

st.set_page_config(page_title="Outbreak Transmission Predictor", page_icon="🦠", layout="wide")

# A little CSS for a cleaner look
st.markdown("""
<style>
.big-result {background:#eaf2ff; border-left:6px solid #2b7de9; padding:16px 20px;
             border-radius:8px; font-size:1.4rem; color:#10213d;}
.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# What each prediction means, in plain English
EXPLAIN = {
    "Person-to-person": "Spread mainly by close contact between people (typical of norovirus in care homes, schools, daycares).",
    "Food": "Spread mainly through contaminated food.",
    "Water": "Spread mainly through water (drinking, pools, hot tubs, lakes).",
    "Animal contact": "Spread mainly through contact with animals.",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---- Header ---------------------------------------------------------------
st.title("🦠 Disease Outbreak Transmission Predictor")
st.caption("College ML project · CDC NORS historical outbreak reports")
st.warning("**Disclaimer:** This is an educational model built from historical data. "
           "It is **not** a real-world public-health warning system and must not be used "
           "for health decisions or to predict whether an outbreak will happen.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. In the terminal run:  `python train_model.py`  "
             "and then restart this app.")
    st.stop()

bundle = load_model()
options = bundle["options"]
metrics = load_metrics() if os.path.exists(METRICS_PATH) else None

tab_predict, tab_perf, tab_about = st.tabs(["🔮 Predict", "📊 Model performance", "ℹ️ About"])

# ---- Tab 1: Predict -------------------------------------------------------
with tab_predict:
    st.subheader("Describe an outbreak report")
    st.write("Given a reported outbreak, the model guesses **how it mainly spread**.")

    c1, c2 = st.columns(2)
    with c1:
        month = st.selectbox("Month", options["Month"], index=6)
        state = st.selectbox("State", options["State"], index=options["State"].index("Ohio")
                             if "Ohio" in options["State"] else 0)
        illnesses = st.number_input("Number of people ill", min_value=1, max_value=20000,
                                    value=20, step=1)
    with c2:
        etiology = st.selectbox("Germ / cause (etiology)", options["Etiology"],
                                index=options["Etiology"].index("Norovirus")
                                if "Norovirus" in options["Etiology"] else 0,
                                help="Choose 'Not reported' if the cause is not known yet.")
        setting = st.selectbox("Where it happened (setting)", options["Setting"],
                               index=options["Setting"].index("Restaurant: Sit-down dining")
                               if "Restaurant: Sit-down dining" in options["Setting"] else 0)

    if st.button("Predict transmission mode", type="primary"):
        best, table = predict_one(bundle, month, state, etiology, setting, illnesses)
        top_prob = table.loc[0, "Probability"]
        st.markdown(f'<div class="big-result">Most likely: <b>{best}</b> '
                    f'({top_prob:.0%} confidence)</div>', unsafe_allow_html=True)
        st.write("")
        st.write(EXPLAIN.get(best, ""))
        if top_prob < 0.6:
            st.info("The model is not very sure about this one - several modes look possible.")
        if best == "Animal contact":
            st.info("Animal contact is the model's weakest category (see the Model performance tab), "
                    "so treat this answer with extra caution.")

        st.subheader("Probability for each mode")
        chart_data = table.set_index("Transmission mode")
        st.bar_chart(chart_data)
        st.dataframe(table.style.format({"Probability": "{:.1%}"}), hide_index=True)

# ---- Tab 2: Performance ---------------------------------------------------
with tab_perf:
    if metrics is None:
        st.info("Run `python train_model.py` to generate performance results.")
    else:
        st.subheader("How well does it work? (on unseen test data)")
        comp = metrics["model_comparison"]
        main = metrics["main_model"]
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{main} accuracy", f"{comp[main]['accuracy']:.1%}")
        m2.metric(f"{main} macro-F1", f"{comp[main]['macro_f1']:.3f}")
        base = "Baseline (always guess most common)"
        m3.metric("Baseline accuracy", f"{comp[base]['accuracy']:.1%}")
        st.caption("Macro-F1 averages the score of all 4 classes equally, so it shows weak "
                   "small classes that plain accuracy hides.")

        st.write("**Model comparison**")
        st.dataframe(pd.DataFrame(comp).T.rename(
            columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1"}).style.format("{:.3f}"))

        st.write("**Per-class results (Random Forest)**")
        per = pd.DataFrame(metrics["per_class"]).T[["precision", "recall", "f1-score", "support"]]
        st.dataframe(per.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                       "f1-score": "{:.2f}", "support": "{:.0f}"}))

        g1, g2 = st.columns(2)
        with g1:
            st.image("reports/confusion_matrix.png")
        with g2:
            st.image("reports/feature_importance.png")
            st.image("reports/class_distribution.png")

# ---- Tab 3: About ---------------------------------------------------------
with tab_about:
    st.subheader("What this project does")
    st.markdown("""
**Task:** multi-class classification - predict an outbreak's *Primary Mode*
(Person-to-person, Food, Water, Animal contact) from the month, state, germ,
setting and number of people ill.

**Why not "will an outbreak happen?"** The NORS file only contains outbreaks that
*already happened*. With no records of "no outbreak", a model cannot learn whether one will occur.

**Data leakage avoided:** columns like *Food Vehicle*, *Water Exposure*, *Water Type*,
*Animal Type* and *IFSAC Category* are only filled in once the mode is known, so they are
**not** used as inputs. Hospitalizations and deaths are later outcomes and are also excluded.
Exact duplicate rows were removed before splitting.

**Limitations**
- The dataset is US-only and reports are not a random sample of all outbreaks.
- The model leans heavily on *setting* and *germ* (e.g. nursing homes + norovirus
  -> person-to-person), so it learns reporting patterns, not biology.
- Animal contact has only ~650 examples, so it is predicted unreliably.
- "Indeterminate/unknown" and "Environmental contamination" modes were dropped.
- Illness counts are only known as the outbreak is investigated, so early use would be rough.
""")
