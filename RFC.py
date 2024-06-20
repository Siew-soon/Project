import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Define feature and target fields
fields_x = [
    "LandForm",
    "SoilTypeAndThickness",
    "Geology",
    "Slope",
    "Aspect",
    "SPI",
    "TWI",
    "STI",
    "Rainfall",
    "DistanceToWaterways",
]
fields_y = ["Class"]

# Load dataset
dataset = pd.read_csv(r"Dataset/FeaturesForAllPoints.csv")


# Encode categorical features and fill missing values
def convert(data):
    encode_data = preprocessing.LabelEncoder()
    data["LandUse"] = encode_data.fit_transform(data.LandUse.astype(str))
    data["LandForm"] = encode_data.fit_transform(data.LandForm.astype(str))
    data["SoilTypeAndThickness"] = encode_data.fit_transform(
        data.SoilTypeAndThickness.astype(str)
    )
    data["Geology"] = encode_data.fit_transform(data.Geology.astype(str))
    data["Slope"] = encode_data.fit_transform(data.Slope.astype(str))
    data["Aspect"] = encode_data.fit_transform(data.Aspect.astype(str))
    data["SPI"] = encode_data.fit_transform(data.SPI.astype(str))
    data["TWI"] = encode_data.fit_transform(data.TWI.astype(str))
    data["STI"] = encode_data.fit_transform(data.STI.astype(str))
    data["Rainfall"] = encode_data.fit_transform(data.Rainfall.astype(str))
    data["DistanceToWaterways"] = encode_data.fit_transform(
        data.DistanceToWaterways.astype(str)
    )
    data = data.fillna(-999)
    return data


dataset = convert(dataset)
X = dataset.loc[:, fields_x]
y = dataset.loc[:, fields_y].values.ravel()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1, shuffle=True
)

# Define and train the classifier
classifier = RandomForestClassifier(
    n_estimators=10,  # Increased number of trees for better generalization
    criterion="entropy",  # Using entropy for information gain
    max_depth=10,  # Limiting the maximum depth to avoid overfitting
    max_features=None,
    min_samples_split=5,  # Increasing min_samples_split for higher node purity
    min_samples_leaf=2,  # Increasing min_samples_leaf for higher node purity
    min_weight_fraction_leaf=0.0,  # Minimum weighted fraction of the total sum of weights for a leaf node
    max_leaf_nodes=17,  # Unlimited maximum leaf nodes
    bootstrap=True,  # Using bootstrap samples
    # n_jobs=-1,                        # Utilizing all processors for parallel processing (CPU)
    random_state=42,  # Setting a random seed for reproducibility
    warm_start=False,  # Not reusing previous solution (Useful when to analysis the performance of different value of estimator with same other parameters value)
    class_weight=None,  # No specific class weights
    ccp_alpha=0.4,  # No complexity pruning
)
classifier.fit(X_train, y_train)

# Predict on training and testing data
y_predTr = classifier.predict(X_train)
y_pred = classifier.predict(X_test)

# Confusion matrices
cm1 = confusion_matrix(y_train, y_predTr)
cm2 = confusion_matrix(y_test, y_pred)
print("Confusion Matrix - Training Data:\n", cm1)
print("Confusion Matrix - Testing Data:\n", cm2)


# Function to plot the ROC curve
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()


# Calculate ROC curve and AUC
probs = classifier.predict_proba(X_test)[:, 1]
fper, tper, _ = roc_curve(y_test, probs)
plot_roc_curve(fper, tper)
auc = roc_auc_score(y_test, probs)
print(f"Area under the curve (AUC): {auc:.4f}")

# Print model accuracy
print(f"Accuracy on Training data: {classifier.score(X_train, y_train):.4f}")
print(f"Accuracy on Testing data: {classifier.score(X_test, y_test):.4f}")


#   Train the classification model using random forest classification
#   Hyperparameter search (Find the best setting)
#   Feature importance (To identiy the important feature)
#   Feature select (Identify selected important feature performance)
#   Data augumentation to increase those data w and w/o feature important
#   Perform recursive feature elimination
#   Data augumentation to increase those data w and w/o feature important
