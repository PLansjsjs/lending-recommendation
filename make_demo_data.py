"""OPTIONAL: create a SYNTHETIC file in Lending Club format to test the pipeline.

The numbers are made up. Use it only to check that everything runs before you
download the real Kaggle data. Never report results from this file.
"""
import numpy as np
import pandas as pd
from common import DATA_DIR

rng = np.random.default_rng(7)
n = 60_000

annual_inc = rng.lognormal(11.1, 0.55, n).round(0)
loan_amnt = (np.clip(rng.lognormal(9.5, 0.6, n), 1000, 40000) / 25).round() * 25
term = rng.choice([36, 60], n, p=[0.72, 0.28])
dti = np.clip(rng.normal(18, 8, n), 0, 45).round(2)
dti[rng.random(n) < 0.001] = 999            # junk values like the real data
revol_util = np.clip(rng.normal(52, 24, n), 0, 120).round(1)
delinq = rng.poisson(0.3, n)
inq = rng.poisson(0.7, n)
open_acc = rng.poisson(11, n) + 2
total_acc = open_acc + rng.poisson(11, n)
pub_rec = rng.binomial(1, 0.12, n)
revol_bal = (annual_inc * rng.uniform(0.02, 0.4, n)).round(0)
home = rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], n, p=[0.40, 0.47, 0.125, 0.005])
verif = rng.choice(["Verified", "Source Verified", "Not Verified"], n, p=[0.32, 0.38, 0.30])
purpose = rng.choice(
    ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
     "small_business", "car", "medical", "wedding", "moving"],
    n, p=[0.46, 0.20, 0.07, 0.08, 0.03, 0.03, 0.03, 0.03, 0.04, 0.03])
emp = rng.choice(["< 1 year", "1 year", "2 years", "3 years", "5 years", "8 years", "10+ years"], n)
emp = np.where(rng.random(n) < 0.06, None, emp)

lti = loan_amnt / annual_inc
risk = (-1.85 + 0.025 * (np.minimum(dti, 60) - 18) + 0.010 * (revol_util - 52)
        + 0.45 * (term == 60) + 1.6 * (lti - 0.22) - 0.55 * (np.log(annual_inc) - 11.1)
        + 0.12 * delinq + 0.07 * inq + 0.25 * pub_rec + rng.normal(0, 0.35, n))
p_default = 1 / (1 + np.exp(-risk))
default = rng.random(n) < p_default

grade_idx = np.clip(np.digitize(risk + rng.normal(0, 0.3, n), np.quantile(risk, [.18, .45, .70, .86, .94, .98])), 0, 6)
grades = np.array(list("ABCDEFG"))[grade_idx]
int_rate = (6 + 2.0 * grade_idx + 1.6 * (term == 60) + rng.normal(0, 0.8, n)).round(2)

status = np.where(default, "Charged Off", "Fully Paid")
status = np.where(rng.random(n) < 0.08, "Current", status)
issue = pd.to_datetime(rng.integers(0, 48, n) * 30 + pd.Timestamp("2013-01-01").value // 86_400_000_000_000, unit="D")

df = pd.DataFrame({
    "loan_amnt": loan_amnt, "term": [f" {t} months" for t in term],
    "int_rate": [f"{r}%" for r in int_rate], "installment": (loan_amnt / term * 1.1).round(2),
    "grade": grades, "sub_grade": [g + str(rng.integers(1, 6)) for g in grades],
    "emp_length": emp, "home_ownership": home, "annual_inc": annual_inc,
    "verification_status": verif, "purpose": purpose, "dti": dti,
    "delinq_2yrs": delinq, "inq_last_6mths": inq, "open_acc": open_acc, "pub_rec": pub_rec,
    "revol_bal": revol_bal, "revol_util": [f"{u}%" for u in revol_util],
    "total_acc": total_acc, "loan_status": status,
    "issue_d": issue.strftime("%b-%Y"),
})
out = DATA_DIR / "demo_loan_synthetic.csv"
df.to_csv(out, index=False)
print(f"Wrote SYNTHETIC demo data: {out}  ({len(df):,} rows). Do not use it for your report.")
