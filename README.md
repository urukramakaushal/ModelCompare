# modelcompare 🏆

Compare 5 machine learning models on any CSV with one command.

It auto-detects classification vs regression, imputes missing values, scales numeric
columns, one-hot encodes text columns, and ranks the models by cross-validated score.

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python modelcompare.py data.csv --target price
python modelcompare.py data.csv --target churned --cv 10
```

## How the task is chosen
A numeric target with more than 10 unique values is treated as regression. Anything else is classification.

## Roadmap
- Add XGBoost / LightGBM
- Save results to CSV
- Feature importance for the best model
- Unit tests

Made by Urukrama. Contributions welcome.
