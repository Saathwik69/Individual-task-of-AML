# Individual-task-of-AML
Comparative Analysis of SVM, Gaussian Naive Bayes and Decision Tree for Diabetes Prediction
Overview
This project presents a comparative machine learning study for diabetes prediction using three classification algorithms:
Support Vector Machine (SVM)
Gaussian Naive Bayes (GNB)
Decision Tree Classifier
The project is based on the Advanced Machine Learning individual task report and demonstrates an end-to-end machine learning workflow: synthetic data generation, exploratory data analysis, preprocessing, model training, evaluation, visualization, interpretability, and prediction.
Academic/educational project: The dataset used by this implementation is synthetic. The model outputs must not be interpreted as medical diagnoses or clinical risk estimates.
Objectives
Generate a reproducible Pima-style synthetic diabetes dataset.
Understand the structure and distribution of the dataset.
Identify invalid zero placeholders in clinical measurements.
Apply leakage-safe missing-value imputation.
Compare SVM, Gaussian Naive Bayes and Decision Tree classifiers.
Evaluate the models using accuracy, precision, recall, F1-score and ROC-AUC.
Analyze confusion matrices and ROC curves.
Visualize the Decision Tree for interpretability.
Demonstrate prediction on a sample patient.
Dataset
The project uses a synthetic dataset containing:
768 records
8 predictor variables
1 binary target variable
Features
Feature
Description
Pregnancies
Number of pregnancies
Glucose
Plasma glucose-style measurement
BloodPressure
Blood-pressure-style measurement
SkinThickness
Skin-fold-style measurement
Insulin
Serum insulin-style measurement
BMI
Body Mass Index
DiabetesPedigreeFunction
Synthetic family-history risk index
Age
Age in years
Outcome
0 = Non-Diabetic, 1 = Diabetic
The dataset structure is inspired by the Pima Indians Diabetes dataset, but the records in this project are synthetically generated.
Preprocessing
The following measurements use zero as a missing-value placeholder:
Glucose
BloodPressure
SkinThickness
Insulin
BMI
These zeros are converted to missing values (NaN) before model training.
Pregnancies is not modified because zero pregnancies is a valid value.
Median imputation is performed inside Scikit-learn pipelines so that imputation statistics are learned from the training data rather than the test data.
Train-Test Split
Training set: 80%
Test set: 20%
random_state = 42
Stratification: Outcome
Machine Learning Models
1. Support Vector Machine
An RBF-kernel SVM is used.
Parameters:
Kernel: rbf
C: 1.5
Gamma: scale
Probability estimates: enabled
Median imputation
StandardScaler
SVM requires feature scaling because its margin and distance calculations are sensitive to feature magnitude.
2. Gaussian Naive Bayes
Gaussian Naive Bayes models the conditional distribution of continuous features within each class.
Configuration:
Median imputation
Default GaussianNB
3. Decision Tree
The Decision Tree is configured to improve generalization and account for class imbalance.
Parameters:
max_depth = 5
min_samples_leaf = 12
class_weight = "balanced"
The tree is also visualized to demonstrate model interpretability.
Evaluation Metrics
The following metrics are calculated:
Accuracy – overall proportion of correct predictions.
Precision – proportion of predicted diabetic cases that are actually diabetic.
Recall – proportion of actual diabetic cases detected by the model.
F1-score – harmonic mean of precision and recall.
ROC-AUC – ability of predicted probabilities to rank positive cases above negative cases.
Because the positive class is smaller than the negative class, accuracy is not considered sufficient by itself.
Results Reported in the Academic Report
The executed workflow documented in the report produced the following test-set results:
Model
Accuracy
Precision
Recall
F1-Score
ROC-AUC
SVM
0.7662
0.5357
0.3947
0.4545
0.7422
Gaussian Naive Bayes
0.7273
0.4545
0.5263
0.4878
0.7672
Decision Tree
0.6883
0.4194
0.6842
0.5200
0.7294
Interpretation
SVM achieved the highest accuracy and precision.
Gaussian Naive Bayes achieved the highest ROC-AUC.
Decision Tree achieved the highest recall and F1-score.
Therefore, there is no single model that dominates every metric.
If identifying more positive cases is the priority, the Decision Tree is the strongest model in this experiment based on F1-score and recall.
These conclusions are limited to this synthetic academic dataset.
Project Structure
.
├── diabetes_project.py
├── diabetes.csv
├── outputs/
│   ├── class_distribution.png
│   ├── feature_histograms.png
│   ├── correlation_heatmap.png
│   ├── svm_confusion_matrix.png
│   ├── gaussian_naive_bayes_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   ├── roc_curve_comparison.png
│   ├── model_comparison.png
│   ├── decision_tree.png
│   └── model_results.csv
└── README.md
diabetes.csv and the outputs/ directory are generated when the Python script is executed.
Requirements
Python 3.9 or newer is recommended.
Install the required libraries:
pip install numpy pandas matplotlib seaborn scikit-learn
Or create a requirements.txt containing:
numpy
pandas
matplotlib
seaborn
scikit-learn
Then install with:
pip install -r requirements.txt
How to Run
Clone the repository:
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
Run the project:
python diabetes_project.py
The script will:
Generate diabetes.csv.
Perform exploratory analysis.
Preprocess the data.
Train all three models.
Print evaluation metrics.
Save plots and results inside outputs/.
Select the model with the highest F1-score.
Run the sample patient prediction.
Sample Patient
The demonstration patient used by the project contains:
Pregnancies = 4
Glucose = 156
BloodPressure = 78
SkinThickness = 34
Insulin = 165
BMI = 34.8
DiabetesPedigreeFunction = 0.72
Age = 46
The academic report describes this patient as being predicted as Diabetic by the Decision Tree with an estimated probability of 0.7342 in the reported execution.
This probability is a model output for the synthetic academic task and must not be used as a medical diagnosis.
Reproducibility
The project uses:
random_state = 42
for reproducibility. Model parameters are fixed so that compatible environments can reproduce the workflow.
Limitations
The dataset is synthetic.
The data-generation process is designed for educational demonstration.
Results should not be generalized to real patient populations.
The models are not clinically validated.
No clinical calibration or external validation is performed.
Accuracy alone should not be used to select a model for screening.
The project does not establish causal relationships between features and diabetes.
Future Enhancements
Possible extensions include:
Logistic Regression
Random Forest
Gradient Boosting
Cross-validation
Hyperparameter optimization
Probability calibration
Decision-threshold tuning
Feature importance analysis
SHAP-based interpretability
Evaluation on a validated real-world dataset
References
Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.
Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20, 273–297.
Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and Regression Trees. Wadsworth.
McKinney, W. (2010). Data structures for statistical computing in Python.
Harris, C. R. et al. (2020). Array programming with NumPy. Nature, 585, 357–362.
Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95.
Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.