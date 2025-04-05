import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten

# Image size and data paths
img_size = (32, 32)
cats_folder = r'C:\Users\DELL\OneDrive\Desktop\SkillCrafttecnologies\Task3\PetImages\Cat'
dogs_folder = r'C:\Users\DELL\OneDrive\Desktop\SkillCrafttecnologies\Task3\PetImages\Dog'

# Load and preprocess images
def load_images(folder_path, label):
    features, labels = [], []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            features.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return features, labels

cat_features, cat_labels = load_images(cats_folder, label=0)
dog_features, dog_labels = load_images(dogs_folder, label=1)

# Combine and convert to arrays
X = np.array(cat_features + dog_features)
y = np.array(cat_labels + dog_labels)

# Visualize class distribution
def plot_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    labels = ['Cats', 'Dogs']
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=['blue', 'orange'])
    plt.title("Class Distribution in Dataset")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

plot_class_distribution(y)

# Show sample images
def plot_sample_images(X, y, num_samples=12):
    plt.figure(figsize=(10, 6))
    indices = random.sample(range(len(X)), num_samples)
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        plt.imshow(X[idx])
        plt.title("Cat" if y[idx] == 0 else "Dog")
        plt.axis("off")
    plt.suptitle("Random Sample Images from Dataset")
    plt.tight_layout()
    plt.show()

plot_sample_images(X, y)

# CNN-based feature extractor
def create_feature_extractor():
    return Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten()
    ])

feature_extractor = create_feature_extractor()

# Extract features
X_features = feature_extractor.predict(X, batch_size=16)

# Apply PCA
n_components = min(X_features.shape[0], X_features.shape[1], 128)
pca = PCA(n_components=n_components)
X_features = pca.fit_transform(X_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"]).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# PCA feature scatter plot
def visualize_pca_features(X_features, y):
    if X_features.shape[1] < 2:
        print("PCA must be at least 2D for visualization.")
        return
    plt.figure(figsize=(8, 8))
    plt.scatter(X_features[y == 0, 0], X_features[y == 0, 1], label='Cats', alpha=0.7, color='blue')
    plt.scatter(X_features[y == 1, 0], X_features[y == 1, 1], label='Dogs', alpha=0.7, color='orange')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Feature Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()

visualize_pca_features(X_features, y)

# Plot prediction confidence
def plot_prediction_confidence(X_test, y_test, model):
    confidence_scores = model.decision_function(X_test)
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores[y_test == 0], bins=20, alpha=0.7, label='Cats', color='blue')
    plt.hist(confidence_scores[y_test == 1], bins=20, alpha=0.7, label='Dogs', color='orange')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Scores')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_prediction_confidence(X_test, y_test, svm_model)

# Class-wise histogram
def plot_class_histogram(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    cat_indices = np.where(y_test == 0)[0]
    dog_indices = np.where(y_test == 1)[0]
    cat_predictions = y_pred[cat_indices]
    dog_predictions = y_pred[dog_indices]
    plt.bar(['True Cats', 'Predicted Cats'], [len(cat_indices), np.sum(cat_predictions == 0)], color='blue')
    plt.bar(['True Dogs', 'Predicted Dogs'], [len(dog_indices), np.sum(dog_predictions == 1)], color='orange')
    plt.ylabel('Count')
    plt.title('Class-Wise Histogram')
    plt.tight_layout()
    plt.show()

plot_class_histogram(y_test, y_pred)

# Probability bar chart for predictions
def plot_prediction_probabilities(X, y_true, model, num_samples=5):
    indices = random.sample(range(len(X)), num_samples)
    X_sample = X[indices]
    y_sample = y_true[indices]
    probs = model.predict_proba(X_sample)
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.bar(["Cat", "Dog"], probs[i], color=["blue", "orange"])
        plt.ylim(0, 1)
        plt.title(f"True: {'Cat' if y_sample[i]==0 else 'Dog'}")
    plt.suptitle("Prediction Probabilities")
    plt.tight_layout()
    plt.show()

plot_prediction_probabilities(X_test, y_test, svm_model)

# SVM decision boundary
def plot_pca_decision_boundary(X, y, model):
    if X.shape[1] != 2:
        print("Decision boundary visualization requires PCA with 2 components.")
        return
    plt.figure(figsize=(8, 6))
    colors = ListedColormap(["#1f77b4", "#ff7f0e"])
    disp = DecisionBoundaryDisplay.from_estimator(model, X, cmap=colors, alpha=0.3)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Cats", c='blue', edgecolor='k', s=30)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Dogs", c='orange', edgecolor='k', s=30)
    plt.title("SVM Decision Boundary on PCA Features")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Optional: visualize decision boundary only if PCA = 2D
if pca.n_components == 2:
    plot_pca_decision_boundary(X_features, y, svm_model)

# Misclassified image viewer
def visualize_misclassified(X, y_test, y_pred, num_samples=5):
    misclassified_indices = np.where(y_test != y_pred)[0]
    plt.figure(figsize=(15, 15))
    for i, index in enumerate(misclassified_indices[:num_samples]):
        img_array = X[index]
        true_label = "Cat" if y_test[index] == 0 else "Dog"
        predicted_label = "Cat" if y_pred[index] == 0 else "Dog"
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_array)
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

visualize_misclassified(X, y_test, y_pred, num_samples=5)

# Predict a single image
def predict_single_image(image_path, feature_extractor, pca, svm_model):
    try:
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        features = feature_extractor.predict(np.expand_dims(img_array, axis=0))
        features_pca = pca.transform(features.reshape(1, -1))
        prediction = "Cat" if svm_model.predict(features_pca)[0] == 0 else "Dog"
        confidence = max(svm_model.predict_proba(features_pca)[0]) * 100

        plt.imshow(img_array)
        plt.title(f"Predicted: {prediction}\nConfidence: {confidence:.2f}%")
        plt.axis("off")
        plt.show()

        return prediction
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error in prediction"

# Predict on a sample image
sample_image_path = r'C:\Users\DELL\OneDrive\Desktop\SkillCrafttecnologies\Task3\PetImages\Dog\1.jpg'
result = predict_single_image(sample_image_path, feature_extractor, pca, svm_model)
print(f"The sample image is predicted as: {result}")
