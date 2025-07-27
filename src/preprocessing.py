import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple

def build_preprocessor(df: pd.DataFrame) -> Tuple[Pipeline, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Prepare preprocessing pipeline and split data."""
    X = df.drop(columns=['loan_approved', 'applicant_id'])
    y = df['loan_approved']

    cat_cols = ['gender', 'caste_category', 'region', 'employment_type']
    num_cols = ['age', 'annual_income', 'loan_amount', 'loan_term_months', 'credit_score', 'existing_loans_count']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return preprocessor, X_train_proc, X_test_proc, y_train, y_test
