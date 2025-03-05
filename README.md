# Cross-Validation Attempt


### Structure of the repository

|File name|Description|
|---|---|
|data|Location to store your data file|
|model|Location to store your estimator files (.pkl), submitted models have been stored here|
|src/data_loader|contains function to extract data|
|src/data_preprocessor|contains all of the preprocessing steps|
|src/main|Main pipeline, execute this file to trigger the pipeline|
|src/model_dt|contains function to train and validate a decision tree model|
|src/model_knn|contains function to train and validate a k-nearest-neighbor model|
|src/model_xgb|contains function to train and validate a xgb model|
|src/model_exporter|contains function to save the train model into model as a .pkl file|


### Customizability 
* In data_loader,
    1) choosing specified data file (pass in a string during creation of DataLoader class)
    2) custom sql syntax to extract information when using `.get_data`
* In data_preprocessor
    1) `.select_features` allows u to choose what feature you want to be included in training
    2) `.clean_data` combines all of the data processing and feature engineering in 1 place
    3) all processing is separated by type of operation (e.g. encoding, outlier removal), look for underscore suffix for the internal method
* In model scripts,
    1) creation of the object requires hyperparameter arguements, adjust as desired
    2) `validate_model` requires X,y and number of folds as arguements
* In model_exporter,
    1) Allows you to name the file to save the .pkkl file

### Flow of pipeline 
##### Data preparation
1. Data is extracted from 'data' folder via _DataLoader_
2. Data is passed onto _DataPreprocessor_ for all feature engineering using `.clean_data`
   * All feature engineering steps are found in data_preprocessor.py
   * Modification for feature engineering should all go under this python script
3. Data at this stage is ready for model training

##### Model Training and evaluation
4. Data from data preparation is selected via `.select_features` using _DataPreprocessor_
5. Features undergo a train/test split.
   * ratio of split and random_state can be chosen here
6. After splitting, data is scaled
7. Model is trained using X_test_scaled
   * use `.evaluate` to get a quick recall score
   * use `.validate_model` to perform a Kfold validation, number of folds is passed in as arguement
9. Model can be exported via `.export_model`using _ModelExporter_
  
|Feature Name|Type of engineering performed|
|-|-|
|ID|Used to remove duplicates|
|Survive|Standardized values to 0 for 'wont survive and 1 for 'will survive', label encoding|
|Gender|Label encoding|
|Smoke|Standardized to 0 and 1, label encoding|To replace with 'No' and 'Yes' accordingly|Possible binary encoding to 0 and 1|
|Diabetes|Ordinal encoding for Normal>Pre-diabetes>Diabetes based on severity|
|Age|Contains negative values within the dataset, used the absolute value assuming the negative values are a result of entry error, use the absolute value is the correct age
|Ejection_Fraction|Standardized 'L' and 'N' with 'Low' and 'Normal', Ordinal Encoding as Low>Normal> High |
|Sodium|Removal of outliers, Scaling to be performed as the number range within dataset is varied|
|Creatinine|Missing values were dropped as proportion is small and log normalization due to skew, Removal of outliers|
|Platelets|Removal of outliers, ,log normalization due to skew, Scaling to be performed as the number range within dataset is varied|
|Creatine_Phosphokinase|Removal of outliers, Scaling to be performed as the number range within dataset is varied|
|Blood_Pressure|Scaling to be performed as the number range within dataset is varied|
|Hemoglobin|Scaling to be performed as the number range within dataset is varied|
|Height|Used tocreate new feature BMI with Weight| 
|Weight|Used to create new feature BMI with Height|

### Summary of EDA
##### Univariate 
* Imbalanced class for target variable at positive 32% vs negative 68%
* Gender Male 65% Female 35%
* Non Smoker 67% Smoker 33%
* Diabetes 20%, Diabetes 21%, Normal 59% 
* Sodium, Creatinine, Platelets, Creatine_Phosphokinase needs to handle outliers
* Creatinine and BMI have differences in terms of positive and negative classes
* Log transformation performed on Multiple columns to reduce the variance 

##### Bivariate
* ~5% of smokers are females
* Ejection Fraction has a trend of higher mortality with increased heart strength
* BMI and Creatinine has noticeable difference between positive and negative classes
* Creatinine and BMI are the only features that seems to have a difference between the positive and negative classes of the target
* Ejection_Fraction has a trend on higher mortality (lower chance of survival) with increasing ejection_fractions.
* Not much insights could be drawn when the continous features were compared to each other.
* Given that this dataset is mild imbalanced (32%) with minority class (positive/survivors), similar ratios are found in Gender,Smoke and Diabetes features.
* Majority of smokers are males, does it have an influence to survival? No, the proportion is similar to the imbalance of the whole dataset
    - An interesting finding, when observations were separated based on Gender and Smoke, only Female smokers had a different ratio, 3x more survivors.
* Are High ejection_fraction observations are they all non-diabetic? Yes all of them are non-diabetic and all did not survive

##### Correlation of features
* Target is positively correlated to Age, Creatinine and BMI
* Target is negatively correlated to Sodium  
* Age is correlated to Creatinine and BMI and target
* Sodium is negatively correlated to target and Creatinine
* Sodium and BMI is negatively correlated
* Creatinie is highly correlated to target variable
* Creatine_Phosphokinase and Hemoglobin are correlated
* BMI is positively correlated to Age, Creatinie, Blood_Pressure
* BMI is negatively correlated to Hemoglobin

### Choosing the correct metric to evaluate models
Firstly before models, a suitable metric should be used. 
* Precision measures ratio of true positives against **all positive predictions** were correct
* Recall measures ratio of true positives against **all positive truths** in dataset
* False Positive = ( Model predicts 1, Truth is 0) (model says patient would survive when they **wont**)
* False Negative = ( Model predicts 0,Truth is 1) (model says patient would not survive when they **will**)
* In the context of the problem, it is more important to know the survival of the patient to administer preemptive treatment.
* A false positive is more costly than a false negative
* Therefore, it is more important to identify patients who will survive correctly. (Low False Positives rate)
* In ML context, this means to observe a **high recall** rate (instead of precision).

### Model Training
1) Firstly, data was split into 60 training, 20 validation and 20 testing. 
2) A base model using Logistic Regression with minimal features from training was selected to reduce training times. (using validation set)
3) Models were first evaluated via accuracy to check for overfitting and underfitting.
4) They were then validated using KFold cross validation in 10 folds. (10 folds as far as i know is industry standard.)
5) 2 more feature sets were tested (1 using all features, the other using features of interest during EDA). The feature set with best score was used
6) Base models of 6 algorithms were used to check base performance. 3 best performing models were selected for tuning
7) Logistic Regression, SVM were not chosen as they performed poorly.
8) Random Forest and XGBoost performed equally, however both were resource intensive, hence 1 of them was not selected.
9) Hyperparameter tuning is performed last as it is the algorithm specific and time consuming. 
10) Hyperparameter tuning was performed individually for KNN, Decision Tree and XGBoost.
11) Once hypermeters were tuned, the configurations were applied back to each models and verified for increased performance (Using validation set)
12) Once improved performance was confirmed, the models were exposed to final testing data for evaluation (3 models submitted in model folder)
