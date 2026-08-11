# 2025ab05294 - Machine-Learning-Assignment-2
BITS Pilani M Tech - Machine Learning Assignment 2

# Credit Card Default Prediction

## Student Information
- Name: SWARUP GHOSH
- Student ID: 2025AB05294
- Course: M.Tech AIML
- Assignment: ML Assignment 2

## (a) Problem Statement
Predict whether a customer will default on the next month's credit card payment.

## (b) Dataset Information
- Dataset: Default of Credit Card Clients
- Total Records: 30000
- Features: 23
- Target: default.payment.next.month

## (c) Github Repository Link
- https://github.com/braveheartuniverse19-maker/Machine-Learning-Assignment-2

## Data Preprocessing
- Removed ID column
- Checked for missing values
- Performed train-test split
- Applied feature scaling where required

## (d) Models Used
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest

## Results Comparison
## Results Comparison

| Model               | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|---------------------|----------|--------|-----------|--------|----------|--------|
| Random Forest       | 0.8120   | 0.7506 | 0.6325    | 0.3580 | 0.4572   | 0.3749 |
| Logistic Regression | 0.8077   | 0.7076 | 0.6868    | 0.2396 | 0.3553   | 0.3244 |
| KNN                 | 0.7928   | 0.7014 | 0.5487    | 0.3564 | 0.4322   | 0.3233 |
| Decision Tree       | 0.7145   | 0.6075 | 0.3694    | 0.4115 | 0.3893   | 0.2042 |
| Naive Bayes         | 0.4160   | 0.6516 | 0.2496    | 0.8176 | 0.3824   | 0.1111 |

## Observations

| ML Model | Observation about model performance |
|----------|----------------------------------------|
| Logistic Regression | Achieved good overall performance with 80.77% accuracy and the highest precision (68.68%) among all models. However, recall was relatively low, meaning many defaulters were missed. |
| Decision Tree | Produced moderate recall (41.15%) but lower accuracy (71.45%) and AUC (0.6075). The model may be prone to overfitting and generalizes less effectively than ensemble methods. |
| KNN | Delivered balanced performance with 79.28% accuracy, 35.64% recall, and 43.22% F1-score. Performance was reasonable but slightly inferior to Random Forest. |
| Naive Bayes | Achieved the highest recall (81.76%), identifying most defaulters, but suffered from very low accuracy (41.60%) and precision (24.96%), resulting in many false positives. |
| Random Forest (Ensemble) | Achieved the best overall performance with the highest accuracy (81.20%), AUC (0.7506), F1-score (0.4572), and MCC (0.3749), providing the best balance between classification metrics. |
| **Overall Winner** | **Random Forest** is the best model because it achieved the highest Accuracy, AUC, F1-score, and MCC, indicating superior overall predictive performance and better handling of the class imbalance compared to the other models. |


MODEL COMPARISON TABLE
                     Accuracy     AUC  Precision  Recall  F1 Score     MCC
Random Forest          0.8120  0.7506     0.6325  0.3580    0.4572  0.3749
Logistic Regression    0.8077  0.7076     0.6868  0.2396    0.3553  0.3244
KNN                    0.7928  0.7014     0.5487  0.3564    0.4322  0.3233
Decision Tree          0.7145  0.6075     0.3694  0.4115    0.3893  0.2042
Naive Bayes            0.4160  0.6516     0.2496  0.8176    0.3824  0.1111

## Observations
ML Model Name                    Observation about model performance
Logistic Regression              Achieved good overall performance with 80.77% accuracy and the highest precision (68.68%) among all models. However, recall was relatively low, meaning many defaulters were missed.
Decision Tree                    Produced moderate recall (41.15%) but lower accuracy (71.45%) and AUC (0.6075). The model may be prone to overfitting and generalizes less effectively than ensemble methods.
KNN                              Delivered balanced performance with 79.28% accuracy, 35.64% recall, and 43.22% F1-score. Performance was reasonable but slightly inferior to Random Forest.
Naive Bayes                      Achieved the highest recall (81.76%), identifying most defaulters, but suffered from very low accuracy (41.60%) and precision (24.96%), resulting in many false positives.
Random Forest (Ensemble)         Achieved the best overall performance with the highest accuracy (81.20%), AUC (0.7506), F1-score (0.4572), and MCC (0.3749), providing the best balance between classification metrics.
Overall Winner for my dataset    Random Forest is the best model because it achieved the highest Accuracy, AUC, F1-score, and MCC, indicating superior overall predictive performance and better handling of the class imbalance compared to the other models.

## Conclusion

Random Forest emerged as the best-performing model, achieving the highest Accuracy (81.20%), AUC (0.7506), F1-score (0.4572), and MCC (0.3749), making it the most balanced and reliable choice for this dataset.
 
The results also highlight a clear precision-recall trade-off driven by class imbalance: Naive Bayes maximized recall (81.76%) but at a heavy cost to accuracy and precision, while Logistic Regression favored precision (68.68%) over recall. Random Forest handled this trade-off best, and has therefore been used as the primary model in the deployed Streamlit app.
