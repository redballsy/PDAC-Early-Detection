# Project Title 
Predict pancreatic ductal adenocarcinoma cancer (PDAC) at the early stage based on biomarkers and patient medical information
## Project Aims
- To predict the patient with PDAC at the earlier stage with a higher accuracy than the previous study on the same dataset with at least 90 %. 
- To find the best machine learning techniques among Random Forest, XG Boost, Decision Tree, SVM that can predict well PDAC.
## Project Objectives
- To Improve the quality of the dataset with feature engineering.
- To combine each model with Grid and random search hyperparameter tuning.
- To Build machine learning models with XG Boost, SVC, Random Forest and Decision Tree that     perform well to get at least an accuracy of 90 %.
## Data
- Data Source: https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer
- Data Description:
  In 2020, an article was published by Silvana Debernardi and colleagues in the journal PLOS Medicine, from a multi-national team of researchers to find and develop 
  an accurate diagnostic test for the most common type of pancreatic cancer, called pancreatic ductal adenocarcinoma or PDAC. They gathered a series of biomarkers 
  from the urine of three groups of patients: Healthy controls, Patients with non-chronic pancreatitis(early stage), Patients with pancreatic ductal adenocarcinoma
- Data Preprocessing:
  I handle the missing values by deleting the first row as it was emptied, and I performed the mean impuation in order to cleanse the missing value from the 
  numerical variables. I also droped the columns sample_id, sample_origin,patient_cohort, diagnosis because they had no influence to predict the stage of the 
  patient. The sample_origine and patient_cohort data are not related to the stages and the diagnosis has constant values.
## Methods
I utilize in this project different machine learning techniques such as XG Boost, SVC, Random Forest and Decision Tree.
  **Model Architecture:**
  
   **The Random Forest** 
    Random Forest classifier was integrated into a machine learning pipeline, and hyperparameter tuning was conducted using a Grid Search CV object with 10-            fold cross-validation which is a technique used to evaluate the performance of a machine learning model and ensure that the model generalizes well to unseen        data. It involves splitting the dataset into 10 equally sized subsets or folds and set parameter for training, validation and calculation of the performance.       Also, the objective of this tuning was to find the optimal set of hyperparameters to maximize classification performance on the training data.
   
   **SVM**
     In the code, the search space for the Support Vector Machine (SVM) was reduced. We put fewer values for the hyperparameters C and gamma to accelerate the           computational load. The parameter grid includes C had a value of [0.1, 1]. Generally, small C give more misclassification but with large margin while larger C      classify perfectly with a risk of overfitting, so the small C parameter was selected to avoid overfitting.  In other hands, Gamma values was 0.1 and 0.01 
     which control the influence of each data point as the lower the values are, it makes the decision boundaries smoother. The RBF kernel was also needed because 
     it is good for data that isn't linearly separable and help the model to find optimal boundary that separate different classes by creating more complex shapes 
     in a higher-dimensional space which is gives a good performance. 
     
   **XGBoost**
     The XG Boost classifier was integrated into a machine learning pipeline, and hyperparameter tuning was conducted using a Grid Search CV object. This technique      was used to evaluate the performance of the XG Boost model and ensure that it generalizes well to unseen data. The parameter grid was defined with different        combinations of n_estimators (number of trees) and learning_rate.
     After defining the grid, the model was trained and validated using cross-validation to identify the best set of hyperparameters. This cross-validation       
     technique, combined with Grid Search, ensures that the model is neither overfitting nor underfitting.
     
   **Decision Tree**
     In the DT code, the hyperparameters tuned was used. The Gini was used to measure the quality of split. The Splitter was used to split at each node, the Max 
     and Min. The depth represented the estimated depth of the tree and finally the minimum leaf was set with acceptable numbers.
     
    **Features**
   Creatinine, LYVE1, REG1B, and TFF1. Age, plasma_CA19_9, stages

## Google Collab

1.  Clone the repository:
   Import the dataset in the google drive

2.  Navigate to the project directory:
    Mount the dataset from Google collab

3.  Installation and import of libraries
    Use python libraries to run the code

## Usage
The code is simple. You just need to upload the the dataset from google drive and import the code from google collab and run it.
