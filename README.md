# ML-Assignment2 : Email Spam Classification

---

## Metadata
| Author          | BITS ID     | Subject                           |
|-----------------|-------------|-----------------------------------|
| Meshach A Martin| 2025AC05804 | S2-25_AIMLCZG565 Machine Learning |

### Live Link: https://ml-assignment2-spambase.streamlit.app/
---

## A. Problem Statement

* **Objective:** Binary classification to detect unwanted spam emails using 57 continuous feature attributes derived from email text characteristics (`1` = Spam, `0` = Ham).
* **Business Impact:** Unfiltered spam degrades productivity, increases infrastructure costs, and introduces security risks such as phishing or malware deployment. Automatic filtering shields end-users while ensuring legitimate communications remain deliverable.
* **Key Challenges:**
  * **Asymmetric Risk:** Misclassifying legitimate email as spam (False Positive) can result in missed critical communications, whereas letting a spam email through (False Negative) is usually a minor inconvenience. A balance between high Precision and high Recall is necessary.
  * **Feature Scaling:** Frequency-based word and character features span varying numerical ranges, requiring normalization for distance and gradient-based models.

---

## B. Dataset Description

* **Dataset Name:** Spambase Dataset
* **Source:** UCI Machine Learning Repository
* **Target Variable:** `target` (`1` = Spam, `0` = Non-Spam / Ham)
* **Total Features:** 57 continuous numerical attributes extracted from processed email text

### Feature Grouping

| Feature Category | Column Count | Description |
|:-----------------|:-------------|:---------------------------------------------------------------------------------------------------------|
| **Word Frequency** | 48 attributes | Percentage of words in the email matching specific keywords (e.g., `make`, `address`, `free`, `credit`, `your`). |
| **Character Frequency** | 6 attributes | Percentage of characters matching specific punctuation (`/`, `[`, `!`, `$`, `#`). |
| **Capital Letters** | 3 attributes | Average, longest, and total count of consecutive capital letters across the text body. |

---

## C. GitHub Repository Link
### https://github.com/meshach-martin/ml-assignment2-spambase

---

## D. Models Used

Five classical supervised classification algorithms representing distinct mathematical paradigms are benchmarked under identical train/test splits:

#### i. Logistic Regression (`LogisticRegression`)
* **Model Type:** Probabilistic Linear Model
* **Description:** Models probability boundaries using a linear combination of input features mapped through a sigmoid function.
* **Data Preprocessing:** Standardized using `StandardScaler` to handle varying feature scales.

#### ii. Decision Tree Classifier (`DecisionTreeClassifier`)
* **Model Type:** Non-Parametric Rule-Based Model
* **Description:** Splits feature space into rectangular regions based on feature thresholds to minimize impurity.
* **Data Preprocessing:** Evaluated on raw feature values without standard scaling.

#### iii. K-Nearest Neighbors (`KNeighborsClassifier`)
* **Model Type:** Distance-Based Classifier
* **Description:** Identifies the $k$-nearest historical data points in multi-dimensional feature space using Euclidean distance to cast majority votes.
* **Data Preprocessing:** Preprocessed using `StandardScaler`.

#### iv. Gaussian Naïve Bayes (`GaussianNB`)
* **Model Type:** Probabilistic Bayesian Classifier
* **Description:** Evaluates class conditional probabilities assuming feature independence under a continuous Gaussian distribution.
* **Data Preprocessing:** Preprocessed using `StandardScaler`.

#### v. Random Forest Classifier (`RandomForestClassifier`)
* **Model Type:** Ensemble Learning (Bagging)
* **Description:** Aggregates predictions across an ensemble of decision trees trained on bootstrapped data subsets.
* **Data Preprocessing:** Evaluated directly on unscaled features.

## Evaluation Metrics Comparision Between Models 

| Model | Accuracy | AUC-ROC | Precision | Recall | F1-Score | MCC |
|:---|:---|:---|:---|:---|:---|:---|
| **Logistic Regression** | **0.9294** | 0.9702 | 0.9209 | 0.8981 | **0.9093** | **0.8518** |
| **Decision Tree** | 0.8990 | 0.9271 | 0.9066 | 0.8292 | 0.8662 | 0.7874 |
| **K-Nearest Neighbors** | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| **Naïve Bayes** | 0.8328 | 0.9376 | 0.7146 | **0.9587** | 0.8188 | 0.6946 |
| **Random Forest** | 0.9197 | **0.9726** | **0.9419** | 0.8485 | 0.8928 | 0.8317 |

## Observations Based On Model Performance

| ML Model Name | Observation about Model Performance |
|:---|:---|
| **Logistic Regression** | **Best overall balanced model.** Achieved the highest Accuracy (92.94%), F1-Score (0.9093), and MCC (0.8518) along with an impressive AUC-ROC of 0.9702. |
| **Random Forest** | **Best classifier for minimizing False Positives.** Delivered the highest Precision (94.19%) and highest overall AUC-ROC (0.9726), making it the safest model against incorrectly flagging legitimate emails as spam. |
| **Naïve Bayes** | **Best classifier for maximizing Recall.** Captured the highest proportion of actual spam emails (Recall of 95.87%), though at the cost of lower Precision (71.46%). |
| **K-Nearest Neighbors** | Demonstrated solid, balanced results across all metrics (Accuracy: 90.77%, F1-Score: 0.8824) after standard scaling. |
| **Decision Tree** | Performed adequately (Accuracy: 89.90%), but lagged behind ensemble and linear counterparts due to high variance on complex boundary separations. |
| ***Overall Winner*** | ***Logistic Regression (Highest Accuracy, F1-Score, and MCC)*** |

---

