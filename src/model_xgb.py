import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import recall_score, make_scorer

class Boost_Model:
    '''This class creates a XGB object 
    Parameter:
    n_depth: max_depth value
    learning_rate: learning rate for this model
    which_metric: which metric to choose
    '''
    def __init__(self, n_depth, learning_rate):
        self.model = xgb.XGBClassifier(max_depth = n_depth, eta = learning_rate)

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