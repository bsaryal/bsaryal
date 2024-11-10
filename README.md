Project 10
Preperation of Datasets
"""
# Import necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# Load the MNIST dataset from TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Display the shape of the training and testing sets
print(f"Training set shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Test set shape: {x_test.shape}, Test labels shape: {y_test.shape}")
# Visualize sample images from each digit class
num_samples = 5
fig, axes = plt.subplots(10, num_samples, figsize=(10, 15))
for digit in range(10):
    digit_images = x_train[y_train == digit]
    for i in range(num_samples):
        ax = axes[digit, i]
        ax.imshow(digit_images[i], cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(f"Digit {digit}")
plt.suptitle("Sample Images from Each Digit Class", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
"""Designing and Building a Deep Learning Model"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Define the CNN model architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
# Display the model architecture
model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
"""Model Training and Monitoring"""
# Train the model and store the history for plotting
batch_size = 32
epochs = 15
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=1
)
# Plot training and validation metrics
plt.figure(figsize=(12, 5))
# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
"""Model Evaluation and Error Analysis
"""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
# Predict on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_classes))
# Display misclassified examples
misclassified_indices = np.where(y_pred_classes != y_test)[0]
# Visualize a few misclassified examples
num_examples = 5
plt.figure(figsize=(10, 10))
for i in range(num_examples):
    index = misclassified_indices[i]
    plt.subplot(1, num_examples, i + 1)
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"True: {y_test[index]}, Pred: {y_pred_classes[index]}")
    plt.axis('off')
plt.suptitle("Examples of Misclassifications", fontsize=16)
