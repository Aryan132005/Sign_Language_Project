"""
train_model.py — Train the gesture classifier on collected landmark data.
Usage: python train_model.py
Outputs: models/sign_model.h5 and models/classes.json
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight as cw

DATA_DIR   = "data/landmarks"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.h5")
CLASS_PATH = os.path.join(MODEL_DIR, "classes.json")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    X, y = [], []
    classes = sorted(os.listdir(DATA_DIR))
    for label in classes:
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith(".npy"):
                landmarks = np.load(os.path.join(label_dir, fname))
                X.append(landmarks)
                y.append(label)
    return np.array(X), np.array(y), classes


def build_model(num_classes):
    model = tf.keras.Sequential([
        layers.Input(shape=(63,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax'),
    ], name="SignLanguageClassifier")
    return model


def train():
    print("Loading dataset...")
    X, y_raw, classes = load_dataset()
    print(f"  {len(X)} samples across {len(classes)} classes: {classes}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(classes))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y)

    weights = cw.compute_class_weight('balanced', classes=np.unique(y),
                                      y=y)
    class_weights = dict(enumerate(weights))

    model = build_model(len(classes))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    cb_list = [
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                                   save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                 restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=7, verbose=1),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1,
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {val_acc:.4f}")

    with open(CLASS_PATH, "w") as f:
        json.dump(classes, f)
    print(f"Saved model  → {MODEL_PATH}")
    print(f"Saved labels → {CLASS_PATH}")


if __name__ == "__main__":
    train()
