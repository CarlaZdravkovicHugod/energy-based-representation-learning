from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.dataloader import Metadata
import os

# TODO: build classifier for the latent space including the metadata


metadataset = Metadata(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'allsup.xlsx')))
df = metadataset.metadata

# Preprocess the metadata
X = df.drop(columns=['sex'])  # Features (drop the target column)
y = df['sex']  # Target column

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert 'sex' to numerical values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from src.comet_models import LatentEBM128  # Replace with the actual EBM import

# Load the pretrained EBM
ebm = LatentEBM128.load('path_to_pretrained_model')  # Replace with the actual path

# Extract features using the EBM
X_train_embeddings = ebm.extract_features(X_train)
X_test_embeddings = ebm.extract_features(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_embeddings, y_train)

# Make predictions
y_pred = clf.predict(X_test_embeddings)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))