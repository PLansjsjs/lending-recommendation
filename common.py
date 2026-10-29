"""Shared paths, settings and helper functions for the Lending Club project.

Every numbered script (00_ ... 06_) imports from this file, so any setting you
want to change (sample size, features, lending assumptions) lives here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"
for _d in (DATA_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = DATA_DIR / "loan_sample.csv"
CLEAN_FILE = DATA_DIR / "loan_clean.csv"

# ------------------------------------------------------------- settings
SEED = 42              # keeps every run reproducible
SAMPLE_SIZE = 50_000   # rows kept from the raw Lending Club file
TEST_SIZE = 0.30       # 70/30 stratified train/test split

# Columns read from the raw Lending Club file.
RAW_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "loan_status", "issue_d",
]
OPTIONAL_RAW = {"grade", "sub_grade", "issue_d", "installment"}

# grade / int_rate are Lending Club's OWN risk assessment (and the interest rate
# is set from the grade). Using them to predict default is circular, so they are
# excluded by default. Set True only to demonstrate the effect in your report.
USE_LC_RISK_FEATURES = False

# ----------------------------------------------- lending-policy assumptions
# These are ASSUMPTIONS, not facts from the data. State them in your report and
# show the sensitivity table produced by 06_recommend.py.
LGD = 0.60            # share of principal lost when a loan defaults
TARGET_MARGIN = 0.03  # minimum expected profit per $1 lent over the loan life
MIN_RATE = 0.05       # lowest interest rate the lender would offer
MAX_RATE = 0.30       # highest rate allowed (usury / affordability cap)
RATE_STEP = 0.0025    # rate search step used when adjusting a rate


# ------------------------------------------------------------- features
def get_feature_lists():
    """Numeric and categorical columns used by the default classifiers."""
    numeric = [
        "loan_amnt", "term", "emp_length", "log_income", "loan_to_income",
        "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
        "log_revol_bal", "revol_util", "total_acc",
    ]
    categorical = ["home_ownership", "verification_status", "purpose"]
    if USE_LC_RISK_FEATURES:
        numeric.append("int_rate")
        categorical.append("grade")
    return numeric, categorical


# Regression (int_rate) predictors
REG_BASE = ["dti", "log_income", "loan_amnt", "term"]
REG_CREDIT = ["revol_util", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec"]
REG_CATEGORICAL = ["verification_status", "home_ownership", "purpose"]

# Clustering features (borrower profile only; no outcome, no interest rate)
CLUSTER_FEATURES = [
    "loan_amnt", "log_income", "dti", "revol_util", "log_revol_bal",
    "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths", "emp_length",
    "pub_rec",
]


def add_derived(df):
    """Create the engineered columns from raw numeric fields."""
    df = df.copy()
    df["log_income"] = np.log(df["annual_inc"].clip(lower=1))
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].clip(lower=1)
    df["log_revol_bal"] = np.log1p(df["revol_bal"].clip(lower=0))
    return df


# ------------------------------------------------------------- data I/O
def load_clean():
    if not CLEAN_FILE.exists():
        raise SystemExit("data/loan_clean.csv not found. Run 00_make_sample.py and 01_clean.py first.")
    return pd.read_csv(CLEAN_FILE)


def get_split(df=None):
    """Stratified train/test split, identical in every script."""
    if df is None:
        df = load_clean()
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["default"], random_state=SEED
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_preprocessor(numeric, categorical, scale=True, drop_first=False):
    """Impute + (scale) numeric columns, impute + one-hot encode categoricals.

    It is used INSIDE sklearn pipelines, so imputation/scaling statistics are
    learned from the training data only (no leakage into the test set).
    """
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    cat_steps = [
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                             drop="first" if drop_first else None)),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline(cat_steps), categorical),
    ])


def reg_matrix(df, medians, columns=None):
    """Numeric design matrix for the regression models.

    columns=None  -> build from scratch (drops the first level of each category)
    columns=[...] -> rebuild with exactly these columns (used when predicting)
    """
    num_cols = REG_BASE + REG_CREDIT
    X = df[num_cols].astype(float).fillna(medians)
    cats = df[REG_CATEGORICAL].astype(str)
    dummies = pd.get_dummies(cats, drop_first=columns is None, dtype=float)
    X = pd.concat([X, dummies], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()
    print(f"  saved figure: outputs/figures/{name}")


def save_table(df, name, index=True):
    df.to_csv(TAB_DIR / name, index=index)
    print(f"  saved table : outputs/tables/{name}")
