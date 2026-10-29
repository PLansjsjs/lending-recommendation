"""Streamlit front end for the Lending Club recommendation model."""
import pandas as pd
import streamlit as st

from common import LGD, TAB_DIR, TARGET_MARGIN
from recommender import load_models, recommend_df

st.set_page_config(page_title="Loan Recommendation Model", page_icon="🏦", layout="wide")

HOME = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIF = ["Verified", "Source Verified", "Not Verified"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
            "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
            "renewable_energy", "educational"]
NUMERIC = ["loan_amnt", "term", "emp_length", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
           "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]
REQUIRED = NUMERIC + ["home_ownership", "verification_status", "purpose"]


@st.cache_resource
def get_models():
    try:
        return load_models()
    except BaseException as e:  # load_models exits if the model files are missing
        st.error(f"Could not load the model files from outputs/models: {e}")
        st.stop()


models = get_models()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Lending assumptions")
st.sidebar.caption("These are assumptions, not facts from the data. Change them to see how the "
                   "recommendation reacts.")
lgd = st.sidebar.slider("Loss given default (share of principal lost)", 0.20, 0.90, float(LGD), 0.05)
target = st.sidebar.slider("Target margin (min expected profit per $1 lent)", 0.00, 0.08,
                           float(TARGET_MARGIN), 0.005)

st.title("🏦 Loan Default Risk and Lending Policy Recommendation")
st.write("Enter a borrower's details to get the predicted default probability, a risk segment and "
         "a recommendation: **Approve**, **Adjust rate** or **Reject**. "
         "Built on Lending Club data (Logistic Regression + multiple regression + K-means).")

tab1, tab2 = st.tabs(["Single borrower", "Upload CSV (many borrowers)"])

# ------------------------------------------------------------ single borrower
with tab1:
    with st.form("borrower"):
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 12000, 500)
            term = st.selectbox("Term (months)", [36, 60])
            annual_inc = st.number_input("Annual income ($)", 5000, 1_000_000, 60000, 1000)
            emp_length = st.slider("Employment length (years)", 0, 10, 4)
            dti = st.number_input("Debt-to-income ratio (%)", 0.0, 60.0, 18.0, 0.5)
        with c2:
            delinq_2yrs = st.number_input("Delinquencies in last 2 years", 0, 20, 0)
            inq_last_6mths = st.number_input("Credit inquiries in last 6 months", 0, 10, 1)
            open_acc = st.number_input("Open credit accounts", 0, 60, 10)
            total_acc = st.number_input("Total credit accounts", 1, 100, 22)
            pub_rec = st.number_input("Public derogatory records", 0, 10, 0)
        with c3:
            revol_bal = st.number_input("Revolving balance ($)", 0, 500_000, 12000, 500)
            revol_util = st.slider("Revolving utilisation (%)", 0.0, 120.0, 50.0, 0.5)
            home_ownership = st.selectbox("Home ownership", HOME)
            verification_status = st.selectbox("Income verification", VERIF)
            purpose = st.selectbox("Loan purpose", PURPOSES)
        go = st.form_submit_button("Get recommendation", type="primary")

    if go:
        row = pd.DataFrame([{
            "loan_amnt": loan_amnt, "term": term, "emp_length": emp_length,
            "annual_inc": annual_inc, "dti": dti, "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths, "open_acc": open_acc, "pub_rec": pub_rec,
            "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
            "home_ownership": home_ownership, "verification_status": verification_status,
            "purpose": purpose}])
        try:
            r = recommend_df(row, models=models, lgd=lgd, target=target).iloc[0]
        except Exception as e:
            st.error(f"Could not score this borrower: {e}")
            st.stop()

        st.subheader("Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Default probability", f"{r['p_default']:.1%}")
        m2.metric("Risk segment", r["segment"])
        m3.metric("Benchmark rate", f"{r['benchmark_rate']:.2%}")
        if r["action"] == "REJECT":
            m4.metric("Recommended rate", "n/a")
            st.error("**REJECT** - no allowed interest rate gives the target expected profit "
                     "for this risk level (or refer for manual review).")
        elif r["action"] == "ADJUST RATE":
            m4.metric("Recommended rate", f"{r['recommended_rate']:.2%}")
            st.warning(f"**ADJUST RATE** - approve only at about {r['recommended_rate']:.2%}, "
                       f"above the {r['benchmark_rate']:.2%} benchmark, to reach the target margin. "
                       f"Expected profit per $1 lent: {r['expected_profit']:.3f}.")
        else:
            m4.metric("Recommended rate", f"{r['recommended_rate']:.2%}")
            st.success(f"**APPROVE** at the benchmark rate. Expected profit per $1 lent: "
                       f"{r['expected_profit']:.3f}.")

# ----------------------------------------------------------------- CSV upload
with tab2:
    st.write("Upload a CSV with these columns (one borrower per row):")
    st.code(", ".join(REQUIRED))
    template = pd.DataFrame([
        [10000, 36, 8, 95000, 9, 0, 0, 9, 0, 8000, 25, 24, "MORTGAGE", "Verified", "home_improvement"],
        [28000, 60, 1, 38000, 33, 2, 4, 14, 1, 22000, 88, 19, "RENT", "Not Verified", "credit_card"],
    ], columns=REQUIRED)
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "borrower_template.csv", "text/csv")
    up = st.file_uploader("Borrowers CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if len(df) > 20000:
                st.warning("Only the first 20,000 rows are scored.")
                df = df.head(20000)
            df = df[REQUIRED].copy()
            for c in NUMERIC:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            try:
                res = recommend_df(df, models=models, lgd=lgd, target=target)
            except Exception as e:
                st.error(f"Could not score the file: {e}")
                st.stop()
            st.subheader("Summary")
            st.bar_chart(res["action"].value_counts())
            show = res[["loan_amnt", "term", "annual_inc", "dti", "p_default", "segment",
                        "benchmark_rate", "action", "recommended_rate", "expected_profit"]]
            st.dataframe(show.style.format({
                "p_default": "{:.1%}", "benchmark_rate": "{:.2%}",
                "recommended_rate": "{:.2%}", "expected_profit": "{:.3f}"}))
            st.download_button("Download results", res.to_csv(index=False),
                               "recommendations.csv", "text/csv")

# -------------------------------------------------------------- segment table
prof = TAB_DIR / "cluster_profile.csv"
if prof.exists():
    with st.expander("What do the risk segments (S1-S6) mean?"):
        st.dataframe(pd.read_csv(prof, index_col=0))

st.caption("Academic project. Default probabilities are statistical estimates from historical "
           "Lending Club data, and LGD and margin are assumptions. Not for real lending decisions.")