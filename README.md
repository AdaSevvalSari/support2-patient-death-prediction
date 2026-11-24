# Predicting Patient Death – SUPPORT2 Dataset

Data Mining term project. The goal is to predict whether a seriously ill hospital patient dies during the follow-up period (`death`: 1 = died, 0 = survived), using information collected when the patient enters the study.

## Dataset
- **Source:** [SUPPORT2 – UCI Machine Learning Repository (#880)](https://archive.ics.uci.edu/dataset/880/support2)
- **Size:** 9,105 patients, 48 columns
- **Target:** `death` – 6,201 died (68.1%), 2,904 survived (31.9%)

## Steps
1. Remove data leakage columns (known only after the outcome)
2. Exploratory data analysis
3. Drop columns with more than 50% missing values
4. Train / test split (80% / 20%, stratified)
5. Outlier treatment with the IQR rule (clip)
6. Fill missing values (median / most frequent)
7. One-hot encoding and standardization
8. Six models tuned with GridSearchCV (5-fold CV): Decision Tree, Naive Bayes, SVM, MLP, Random Forest, Logistic Regression
9. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Specificity, MCC, Cohen's Kappa, Balanced Accuracy

## Results (test set)
| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | 0.816 | 0.759 | 0.834 |
| SVM | 0.812 | 0.751 | 0.839 |
| Random Forest | 0.810 | 0.754 | 0.836 |
| Naive Bayes | 0.760 | 0.666 | 0.709 |
| Decision Tree | 0.758 | 0.739 | 0.820 |
| MLP | 0.743 | 0.714 | 0.799 |

## How to run
```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook SUPPORT2_Project.ipynb
```
The notebook reads the data from `data/support2.csv`.
