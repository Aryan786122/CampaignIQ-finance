"""Feature definition and leakage-safe preprocessing pipelines for CampaignIQ.

Categorical and numerical features are identified and processed using standard
scikit-learn ColumnTransformer and Pipeline objects.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Explicit pre-contact feature subsets
NUMERIC_FEATURES = [
    "age",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


def create_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer for preprocessing numeric and categorical features.

    - Numeric: Median imputation followed by standard scaling.
    - Categorical: Constant imputation ('unknown') followed by OneHotEncoding with
      handle_unknown='ignore' to gracefully handle novel or unseen categories at inference.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retrieve output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        num_names = NUMERIC_FEATURES.copy()
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return num_names + cat_names
