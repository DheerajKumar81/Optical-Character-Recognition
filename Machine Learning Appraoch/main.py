import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def run_mnist_svm_ocr():
    # 1. Load the MNIST dataset from OpenML
    print("Step 1: Fetching MNIST dataset (784 features per image)...")
    # as_frame=False ensures data is returned as NumPy arrays rather than a Pandas DataFrame
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    
    # Pixel values range from 0 to 255. Normalize them to [0, 1] for better SVM convergence.
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    
    print(f"Dataset successfully loaded. Total samples: {X.shape[0]}")

    # 2. Split into Training and Testing Subsets
    # We use a subset of 10,000 training samples to keep CPU training times low (under 1-2 minutes)
    # stratify=y ensures balanced class distributions across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=10000, test_size=2000, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # 3. Initialize the SVM Classifier
    # We use the RBF (Radial Basis Function) kernel, which maps pixels into a non-linear space.
    # C=5.0 provides high regularization to balance margin maximization and classification errors.
    print("\nStep 2: Training the SVM classifier (RBF kernel)...")
    classifier = svm.SVC(kernel='rbf', C=5.0, gamma='scale')
    
    # Train the model
    classifier.fit(X_train, y_train)
    print("Training complete!")

    # 4. Evaluate the Model on Test Data
    print("\nStep 3: Evaluating performance on unseen test data...")
    predictions = classifier.predict(X_test)

    # 5. Generate and Display Performance Metrics
    print("\n================ CLASSIFICATION REPORT ================")
    print(metrics.classification_report(y_test, predictions))

    print("================ CONFUSION MATRIX ================")
    conf_matrix = metrics.confusion_matrix(y_test, predictions)
    print(conf_matrix)

    accuracy = metrics.accuracy_score(y_test, predictions)
    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

    # 6. Visualize Sample Predictions from the Input Image Set
    print("\nStep 4: Displaying sample predictions visual window...")
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        # Reshape the flat 784 element array back into a 28x28 pixel image
        image = X_test[i].reshape(28, 28)
        
        ax.set_axis_off()
        # Display image using a reversed grayscale colormap
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        
        # Color title based on correct/incorrect prediction
        title_text = f"Pred: {predictions[i]}\nTrue: {y_test[i]}"
        if predictions[i] == y_test[i]:
            ax.set_title(title_text, color='green')
        else:
            ax.set_title(title_text, color='red')

    plt.suptitle("SVM MNIST Digit Recognition Results", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_mnist_svm_ocr()

