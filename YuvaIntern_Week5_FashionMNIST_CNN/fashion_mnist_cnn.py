"""
YuvaIntern Week 5 - Deep Learning Application in Data Science
Project: Fashion-MNIST Image Classification using CNN
Framework: TensorFlow / Keras

Run:
    pip install -r requirements.txt
    python fashion_mnist_cnn.py

The script generates:
- model_summary.txt
- training_history.csv
- training_curves.png
- confusion_matrix.png
- classification_report.txt
- sample_predictions.png
- final_model.keras
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("outputs", exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 1. Load publicly available Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# 2. Preprocessing
# Normalize pixels from [0,255] to [0,1] and add channel dimension.
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Hold out 10% of the training set for validation.
val_size = 6000
x_val, y_val = train_images[-val_size:], train_labels[-val_size:]
x_train, y_train = train_images[:-val_size], train_labels[:-val_size]

print("Training samples:", len(x_train))
print("Validation samples:", len(x_val))
print("Test samples:", len(test_images))

# 3. CNN architecture
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.40),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

with open("outputs/model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# 4. Training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 5. Save history
history_df = pd.DataFrame(history.history)
history_df.to_csv("outputs/training_history.csv", index=False)

# 6. Training curves
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_accuracy.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_loss.png", dpi=200)
plt.close()

# Combined graph for the report
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

fig.tight_layout()
fig.savefig("outputs/training_curves.png", dpi=200)
plt.close(fig)

# 7. Test evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

# 8. Predictions and classification metrics
probabilities = model.predict(test_images, batch_size=256, verbose=0)
predictions = np.argmax(probabilities, axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, predictions, average="weighted", zero_division=0
)

report = classification_report(
    test_labels,
    predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

with open("outputs/classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Fashion-MNIST CNN Classification Report\n")
    f.write("=" * 55 + "\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1-score: {f1:.4f}\n\n")
    f.write(report)

# 9. Confusion matrix
cm = confusion_matrix(test_labels, predictions)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title("Fashion-MNIST CNN Confusion Matrix")
fig.tight_layout()
fig.savefig("outputs/confusion_matrix.png", dpi=200)
plt.close(fig)

# 10. Sample predictions
rng = np.random.default_rng(SEED)
indices = rng.choice(len(test_images), 12, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for ax, idx in zip(axes.ravel(), indices):
    ax.imshow(test_images[idx].squeeze(), cmap="gray")
    predicted_name = CLASS_NAMES[predictions[idx]]
    actual_name = CLASS_NAMES[test_labels[idx]]
    ax.set_title(f"Pred: {predicted_name}\nTrue: {actual_name}", fontsize=9)
    ax.axis("off")

fig.tight_layout()
fig.savefig("outputs/sample_predictions.png", dpi=200)
plt.close(fig)

# 11. Save trained model
model.save("outputs/fashion_mnist_cnn.keras")

print("\nFinal test metrics")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nAll report artifacts are in the outputs/ folder.")
