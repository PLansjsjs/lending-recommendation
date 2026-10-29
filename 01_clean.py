"""Step 1 (Module 1) - data cleansing and preparation.

Input : data/loan_sample.csv     Output: data/loan_clean.csv + cleaning tables
Imputation and scaling are NOT done here; they happen inside the model pipelines
so that they are learned from the training set only.
"""
import re

import numpy as np
import pandas as pd
from common import CLEAN_FILE, SAMPLE_FILE, add_derived, save_table

# Variable dictionary used for the "data types and scales" table in the report.
DICTIONARY = {
    "default": ("Target (0 = Fully Paid, 1 = Charged Off)", "Binary", "Nominal"),
    "loan_amnt": ("Loan amount requested ($)", "Numeric", "Ratio"),
    "term": ("Loan term in months (36 / 60)", "Numeric", "Ratio"),
    "int_rate": ("Interest rate (%)", "Numeric", "Ratio"),
    "grade": ("Lending Club risk grade A-G", "Categorical", "Ordinal"),
    "emp_length": ("Employment length (years, 0-10)", "Numeric", "Ratio"),
    "home_ownership": ("Home ownership status", "Categorical", "Nominal"),
    "annual_inc": ("Annual income ($)", "Numeric", "Ratio"),
    "verification_status": ("Income verification status", "Categorical", "Nominal"),
    "purpose": ("Loan purpose", "Categorical", "Nominal"),
    "dti": ("Debt-to-income ratio (%)", "Numeric", "Ratio"),
    "delinq_2yrs": ("Delinquencies in past 2 years", "Numeric (count)", "Ratio"),
    "inq_last_6mths": ("Credit inquiries in last 6 months", "Numeric (count)", "Ratio"),
    "open_acc": ("Open credit lines", "Numeric (count)", "Ratio"),
    "pub_rec": ("Derogatory public records", "Numeric (count)", "Ratio"),
    "revol_bal": ("Revolving balance ($)", "Numeric", "Ratio"),
    "revol_util": ("Revolving utilisation (%)", "Numeric", "Ratio"),
    "total_acc": ("Total credit lines", "Numeric (count)", "Ratio"),
    "loan_to_income": ("Engineered: loan_amnt / annual_inc", "Numeric", "Ratio"),
    "log_income": ("Engineered: log of annual income", "Numeric", "Ratio"),
    "log_revol_bal": ("Engineered: log(1 + revolving balance)", "Numeric", "Ratio"),
}


def to_number(series):
    """'13.5%' -> 13.5 ; works for plain numbers too."""
    s = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def parse_emp_length(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    if text.startswith("<"):
        return 0.0
    found = re.search(r"\d+", text)
    return float(found.group()) if found else np.nan


def main():
    if not SAMPLE_FILE.exists():
        raise SystemExit("data/loan_sample.csv not found. Run 00_make_sample.py first.")
    df = pd.read_csv(SAMPLE_FILE, low_memory=False)
    log = []

    def step(name, before, after, note=""):
        log.append({"step": name, "rows_before": before, "rows_after": after, "note": note})
        print(f"  {name}: {before:,} -> {after:,}  {note}")

    # 1. keep finished loans only and build the target
    n = len(df)
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)
    step("Keep Fully Paid / Charged Off", n, len(df), "'Current' loans have no outcome yet")

    # 2. remove duplicates
    n = len(df)
    df = df.drop_duplicates()
    step("Drop duplicate rows", n, len(df))

    # 3. convert text columns to numbers
    df["term"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["int_rate"] = to_number(df["int_rate"])
    df["revol_util"] = to_number(df["revol_util"])
    df["emp_length"] = df["emp_length"].apply(parse_emp_length)
    if "issue_d" in df:
        df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year

    # 4. categorical tidy-up
    df["home_ownership"] = df["home_ownership"].replace({"ANY": "OTHER", "NONE": "OTHER"})
    share = df["purpose"].value_counts(normalize=True)
    rare = share[share < 0.01].index
    df["purpose"] = df["purpose"].where(~df["purpose"].isin(rare), "other")

    # 5. impossible / extreme values
    n = len(df)
    df = df[(df["annual_inc"] > 0) & df["loan_amnt"].notna() & df["term"].notna() & df["int_rate"].notna()]
    step("Drop missing/zero income, amount, term, rate", n, len(df))
    bad_dti = (df["dti"] < 0) | (df["dti"] > 100)
    bad_util = (df["revol_util"] < 0) | (df["revol_util"] > 150)
    df.loc[bad_dti, "dti"] = np.nan
    df.loc[bad_util, "revol_util"] = np.nan
    print(f"  Set {int(bad_dti.sum())} impossible dti and {int(bad_util.sum())} impossible revol_util values to missing")

    # 6. winsorise heavy right tails at the 99th percentile
    for col in ["annual_inc", "revol_bal"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)
        print(f"  Capped {col} at the 99th percentile ({cap:,.0f})")

    # 7. engineered features
    df = add_derived(df)

    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved {len(df):,} rows x {df.shape[1]} columns to {CLEAN_FILE.name}")
    print(f"Default rate: {df['default'].mean():.1%}")

    # tables for the report
    save_table(pd.DataFrame(log), "cleaning_log.csv", index=False)
    miss = df.isna().mean().mul(100).round(2).rename("missing_%").to_frame()
    save_table(miss[miss["missing_%"] > 0].sort_values("missing_%", ascending=False), "missing_values.csv")
    dic = pd.DataFrame(
        [(v, *DICTIONARY[v]) for v in DICTIONARY if v in df.columns],
        columns=["variable", "description", "data_type", "scale"])
    save_table(dic, "data_types_scales.csv", index=False)


if __name__ == "__main__":
    main()
