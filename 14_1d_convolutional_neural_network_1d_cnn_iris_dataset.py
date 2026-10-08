import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names
num_classes = len(class_names)

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [class_names[i] for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

y_train_encoded = to_categorical(y_train, num_classes)
y_test_encoded = to_categorical(y_test, num_classes)

inputs = Input(shape=(X_train_cnn.shape[1], 1), name="iris_input")
x = Conv1D(filters=32, kernel_size=2, padding="same", use_bias=False, name="conv_block_1")(inputs)
x = BatchNormalization(name="batch_norm_1")(x)
x = Activation("relu", name="relu_1")(x)

x = Conv1D(filters=64, kernel_size=2, padding="same", use_bias=False, name="conv_block_2")(x)
x = BatchNormalization(name="batch_norm_2")(x)
x = Activation("relu", name="relu_2")(x)

x = Conv1D(filters=64, kernel_size=2, padding="same", use_bias=False, name="conv_block_3")(x)
x = BatchNormalization(name="batch_norm_3")(x)
x = Activation("relu", name="relu_3")(x)

max_features = GlobalMaxPooling1D(name="global_max_pooling")(x)
avg_features = GlobalAveragePooling1D(name="global_average_pooling")(x)
x = Concatenate(name="feature_fusion")([max_features, avg_features])

x = Dense(64, activation="relu", name="dense_1")(x)
x = Dropout(0.30, name="dropout_1")(x)
x = Dense(32, activation="relu", name="dense_2")(x)
x = Dropout(0.20, name="dropout_2")(x)
outputs = Dense(num_classes, activation="softmax", name="class_probabilities")(x)

model = Model(inputs=inputs, outputs=outputs, name="Advanced_Iris_1D_CNN")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(filepath="best_iris_cnn.keras", monitor="val_loss", save_best_only=True, verbose=1)

history = model.fit(
    X_train_cnn, y_train_encoded,
    epochs=50, batch_size=16,
    validation_split=0.20,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1, shuffle=True
)

test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_encoded, verbose=0)
y_prob = model.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\nFINAL TEST RESULTS")
print(f"Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names, digits=4))
