import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = fashion_mnist.load_data()
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
num_classes = len(class_names)

X_train_norm = X_train_raw.astype("float32") / 255.0
X_test_norm = X_test_raw.astype("float32") / 255.0

X_train_2d = np.expand_dims(X_train_norm, axis=-1)
X_test_2d = np.expand_dims(X_test_norm, axis=-1)

y_train_encoded = to_categorical(y_train_raw, num_classes)
y_test_encoded = to_categorical(y_test_raw, num_classes)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_2d, y_train_encoded, test_size=0.15, random_state=SEED, stratify=y_train_raw
)

inputs = Input(shape=(28, 28, 1), name="image_input")
x = Conv2D(filters=32, kernel_size=(3, 3), padding="same", use_bias=False, name="conv_block_1")(inputs)
x = BatchNormalization(name="bn_1")(x)
x = Activation("relu", name="relu_1")(x)
x = MaxPooling2D(pool_size=(2, 2), name="max_pool_1")(x)

x = Conv2D(filters=64, kernel_size=(3, 3), padding="same", use_bias=False, name="conv_block_2")(x)
x = BatchNormalization(name="bn_2")(x)
x = Activation("relu", name="relu_2")(x)
x = MaxPooling2D(pool_size=(2, 2), name="max_pool_2")(x)

x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", use_bias=False, name="conv_block_3")(x)
x = BatchNormalization(name="bn_3")(x)
x = Activation("relu", name="relu_3")(x)

max_pool = GlobalMaxPooling2D(name="global_max_pool")(x)
avg_pool = GlobalAveragePooling2D(name="global_avg_pool")(x)
x = Concatenate(name="feature_fusion")([max_pool, avg_pool])

x = Dense(128, activation="relu", name="dense_1")(x)
x = Dropout(0.30, name="dropout_1")(x)
x = Dense(64, activation="relu", name="dense_2")(x)
x = Dropout(0.20, name="dropout_2")(x)
outputs = Dense(num_classes, activation="softmax", name="softmax_output")(x)

model = Model(inputs=inputs, outputs=outputs, name="Advanced_2D_Image_CNN")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath="best_2d_cnn.keras", monitor="val_loss", save_best_only=True, verbose=1)
]

history = model.fit(
    X_train_split, y_train_split,
    epochs=10, batch_size=64,
    validation_data=(X_val_split, y_val_split),
    callbacks=callbacks,
    verbose=1, shuffle=True
)

test_loss, test_accuracy = model.evaluate(X_test_2d, y_test_encoded, verbose=0)
y_prob = model.predict(X_test_2d, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print(f"2D CNN Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test_raw, y_pred, target_names=class_names, digits=4))
