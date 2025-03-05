from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import recall_score, make_scorer

class DT_Model:
    '''This class creates a Decision Tree object 
    Parameter:
    n_sample_split: value for min_sample_split
    n_features: value for max_features 
    n_depth : value for max_depth
    '''
    def __init__(self, n_sample_split, n_features, n_depth):
        self.model = DecisionTreeClassifier(min_samples_split = n_sample_split, max_features = n_features,  max_depth = n_depth)

    def train(self, X_train, y_train):
        '''This fuction serves to fit model with training data
        Parameter:
        X_train: Features of Training Data
        y_train: Outcomes of Training Data

        Returns:
        Model trained with X_Train and y_train
        '''
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        '''This function will pass in features of testing data to obtain predictions
        Parameter:
        X_test: features of the testing data
        
        Returns:
        predictions of the testing data'''
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        '''This function will provide the recall score when prediction of the model is compared to the ground truth in testing data
        Parameter
        X_test: features of testing data (used to obtain predictions)
        y_test: outcomes in testing data to compare against predictions

        Returns:
        Recall score
        '''
        y_pred = self.predict(X_test)
        recall = recall_score(y_test, y_pred)
        return recall
    
    def validate_model(self,X_data, y_data, folds):
        '''This function will enable a KFold cross validation in the desired metric
        Parameter:
        self.model: The trained model (estimator)
        X_data: the feature data
        y_dat: the outcome data
        folds: number of folds to be used for cross validation
        score: the scoring metric of choice
        
        Returns:
        An average score of the results from n_folds
        '''
        metric = make_scorer(recall_score)
        kf = KFold(n_splits = folds, shuffle = True)
        cv_results = cross_val_score(self.model, X_data, y_data, cv=kf, scoring=metric) 
        print('The average score of ' + str(folds) + ' fold cross validation is ' + str(cv_results.mean()))