"""modelcompare: compare several ML models on any CSV with one command."""
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MODELS = {
    "classification": {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
        "KNN": KNeighborsClassifier(),
    },
    "regression": {
        "Ridge": Ridge(),
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Random Forest": RandomForestRegressor(random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=0),
        "KNN": KNeighborsRegressor(),
    },
}
METRIC = {"classification": "accuracy", "regression": "r2"}


def detect_task(y: pd.Series) -> str:
    numeric = pd.api.types.is_numeric_dtype(y)
    return "regression" if numeric and y.nunique() > 10 else "classification"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat),
    ])


def compare(df: pd.DataFrame, target: str, cv: int = 5) -> pd.DataFrame:
    df = df.dropna(subset=[target])
    X, y = df.drop(columns=[target]), df[target]
    task = detect_task(y)
    rows = []
    for name, model in MODELS[task].items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=METRIC[task])
        rows.append({"model": name, METRIC[task]: round(scores.mean(), 4), "std": round(scores.std(), 4)})
    out = pd.DataFrame(rows).sort_values(METRIC[task], ascending=False).reset_index(drop=True)
    out.attrs["task"] = task
    return out


def main():
    p = argparse.ArgumentParser(description="Compare ML models on a CSV file.")
    p.add_argument("csv", help="path to CSV file")
    p.add_argument("--target", required=True, help="name of the column to predict")
    p.add_argument("--cv", type=int, default=5, help="cross-validation folds (default 5)")
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    if a.target not in df.columns:
        raise SystemExit(f"Column '{a.target}' not found. Available: {', '.join(df.columns)}")
    res = compare(df, a.target, a.cv)
    print(f"\nTask: {res.attrs['task']} | {a.cv}-fold cross-validation\n")
    print(res.to_string(index=False))
    print(f"\n🏆 Best model: {res.loc[0, 'model']}")


if __name__ == "__main__":
    main()
