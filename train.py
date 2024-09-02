# train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os


# Define S3 bucket and file details
bucket_name = 'datasetscybersec'
file_key = 'NID/Train_data.csv'

# Initialize a session using Amazon S3
s3 = boto3.client('s3')

# Download the file from S3 to a temporary file
with NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
    s3.download_fileobj(bucket_name, file_key, temp_file)
    temp_file_path = temp_file.name

# Load data from the temporary CSV file
data = pd.read_csv(temp_file_path)

# Assume your target column is named 'target' and all other columns are features
X = data.drop(columns='class')  # Replace 'target' with the actual target column name
y = data['class']  # Replace 'target' with the actual target column name

# Load Iris dataset
# data = load_iris()
# X, y = data.data, data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model to the output directory
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
joblib.dump(clf, os.path.join(model_dir, "model.joblib"))


