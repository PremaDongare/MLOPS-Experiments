from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# Option 1: Use local file storage (recommended for beginners)
# MLflow will create a 'mlruns' folder in your current directory
# mlflow.set_tracking_uri("file:./mlruns")

# Option 2: Use MLflow server (make sure to start server first)
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()  # builtin dataset
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 10

# Mention your experiment below
mlflow.set_experiment('wine_classification_experiment')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics and parameters
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    
    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # Save plot
    plt.savefig("confusion_matrix.png")
    
    # Log artifacts using mlflow
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    
    # Close the plot to free memory
    plt.close()
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Run completed successfully!")