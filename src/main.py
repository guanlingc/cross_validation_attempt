from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split
from model_knn import KNN_Model
from model_dt import DT_Model
from model_xgb import Boost_Model
from model_exporter import ModelExporter


# Data extraction from source file
data = DataLoader("data/survive.db")                    # Creates the object and passing in file path as a string
data.check_data_path()                                  # Returns your data path for confirmation
conn = data.initiate_local_connection()                 # Attempts to connect to source file
df = data.get_data(f'SELECT * FROM survive', conn)      # Pass in SQL syntax as a string and stores information as pandas dataframe df

# Preprocessing of data
data_processor = DataPreprocessor()                       # Create the object 
df = data_processor.clean_data(df)                        # Cleans the data

# Selection of data for X and Y
X, y = data_processor.select_features(df)

# Splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scaling of data
X_train_scaled, X_test_scaled = data_processor.scale_data(X_train,X_test)

# Model Training for KNN
knn = KNN_Model(10, 'distance', 'manhattan')
knn.train(X_train_scaled, y_train)
knn_recall = knn.evaluate(X_test_scaled, y_test)
print(f'The recall of this KNN model is ', knn_recall)

# # Model Training for DT
dt = DT_Model(7, 5, 12)
dt.train(X_train_scaled, y_train)
dt_recall = dt.evaluate(X_test_scaled, y_test)
print(f'The recall of this Decision Tree model is ', dt_recall)

# Model Training for XGB
boost = Boost_Model(3, 0.5)
boost.train(X_train_scaled, y_train)
boost_recall = boost.evaluate(X_test_scaled, y_test)
print(f'The recall of this XGB model is ', boost_recall)

# Validate model using Kfold
dt.validate_model(X_train_scaled, y_train, 10)
boost.validate_model(X_train_scaled, y_train, 10)
knn.validate_model(X_train_scaled, y_train, 10)

# To export models
exporter = ModelExporter()
exporter.export_model(boost, 'xgboost_model_depth3_learning0.5')