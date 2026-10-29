"""Try the recommender on individual borrowers (the 'software' demo for your DA).

Edit the numbers below, then run:  python demo_borrower.py
Needs steps 03, 04 and 05 to have been run first (python run_all.py does this).
"""
import pandas as pd
from recommender import recommend_df

BORROWERS = [
    {"name": "Applicant 1 - steady income, low debt", "loan_amnt": 10000, "term": 36, "emp_length": 8,
     "annual_inc": 95000, "dti": 9.0, "delinq_2yrs": 0, "inq_last_6mths": 0, "open_acc": 9, "pub_rec": 0,
     "revol_bal": 8000, "revol_util": 25.0, "total_acc": 24, "home_ownership": "MORTGAGE",
     "verification_status": "Verified", "purpose": "home_improvement"},
    {"name": "Applicant 2 - average profile", "loan_amnt": 15000, "term": 36, "emp_length": 4,
     "annual_inc": 62000, "dti": 19.0, "delinq_2yrs": 0, "inq_last_6mths": 1, "open_acc": 11, "pub_rec": 0,
     "revol_bal": 14000, "revol_util": 55.0, "total_acc": 22, "home_ownership": "RENT",
     "verification_status": "Source Verified", "purpose": "debt_consolidation"},
    {"name": "Applicant 3 - stretched borrower", "loan_amnt": 28000, "term": 60, "emp_length": 1,
     "annual_inc": 38000, "dti": 33.0, "delinq_2yrs": 2, "inq_last_6mths": 4, "open_acc": 14, "pub_rec": 1,
     "revol_bal": 22000, "revol_util": 88.0, "total_acc": 19, "home_ownership": "RENT",
     "verification_status": "Not Verified", "purpose": "credit_card"},
]


def main():
    df = pd.DataFrame(BORROWERS)
    res = recommend_df(df.drop(columns="name"))
    for name, (_, r) in zip(df["name"], res.iterrows()):
        print("=" * 62)
        print(name)
        print(f"  Predicted probability of default : {r['p_default']:.1%}")
        print(f"  Risk segment                     : {r['segment']}")
        print(f"  Benchmark rate for this profile  : {r['benchmark_rate']:.2%}")
        print(f"  RECOMMENDATION                   : {r['action']}")
        if r["action"] == "REJECT":
            print("  Reason: no allowed rate gives the target expected profit.")
        else:
            print(f"  Recommended interest rate        : {r['recommended_rate']:.2%}")
            print(f"  Expected profit per $1 lent      : {r['expected_profit']:.3f}")


if __name__ == "__main__":
    main()
