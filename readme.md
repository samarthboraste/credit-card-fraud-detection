# Credit Card Fraud Detection

Machine learning project to detect fraudulent credit card transactions on a highly imbalanced dataset (about 0.17% fraud), using SMOTE to balance the training data and comparing several classification models.

## Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) – European cardholder transactions over two days.

- 284,807 transactions, 31 columns
- `V1`–`V28`: anonymized PCA features
- `Time`, `Amount`, and `Class` (0 = genuine, 1 = fraud)
- Only 492 frauds (about 0.17%)

**The dataset is not included in this repository.** Download `creditcard.csv` from the link above and place it in the project folder.

## Project Workflow

1. **Data inspection:** shape, class distribution, duplicate rows (1,081 found and removed)
2. **EDA:** class imbalance plot, correlation of features with `Class`, `Time` distribution by class
3. **Preprocessing:**
   - `Amount` scaled with `StandardScaler` (`Normalized_Amount`)
   - `Time` dropped: it is only seconds since the first transaction, has very low correlation with fraud, overlaps heavily between classes, and would not generalize
4. **Train/test split:** stratified, so both sets keep the same fraud ratio
5. **Balancing:** SMOTE applied to the **training data only**; the test set stays untouched
6. **Modeling and evaluation:** metrics on the untouched test set (56,656 genuine, 90 fraud)

## Results (fraud class)

| Model | Precision | Recall | F1-score |
|---|---|---|---|
| Decision Tree | 0.36 | 0.71 | 0.48 |
| **Random Forest** | **0.92** | **0.79** | **0.85** |
| XGBoost | 0.79 | 0.78 | 0.78 |
| Logistic Regression | 0.06 | 0.89 | 0.10 |

Accuracy is about 99–100% for most models because 99.8% of transactions are genuine, so it is not a useful metric here. Models were compared using fraud precision, recall and F1.

## Conclusion

Random Forest gave the best balance of precision (0.92) and recall (0.79), with an F1-score of 0.85, so it was selected as the final model. XGBoost was close behind. Logistic Regression had the highest recall but flagged far too many genuine transactions (precision 0.06), which makes it impractical.

## Limitations

- The test set has only 90 fraud cases, so scores can shift by a few points with a different split.
- `Amount` was scaled before the train/test split, which causes minor data leakage.
- Results are limited to the models tested; LightGBM and hyperparameter tuning were not explored.
- The features `V1`–`V28` are anonymized, so the model cannot be interpreted in business terms.

## Future Work

- Hyperparameter tuning
- Try LightGBM
- Decision threshold tuning
- Simple Streamlit demo

## How to Run

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle into this folder, then open and run:

```bash
jupyter notebook