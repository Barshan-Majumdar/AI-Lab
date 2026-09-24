import io
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import ipywidgets as widgets

mpl.rcParams["animation.embed_limit"] = 100

print("=" * 60)
print("                  KNN CLASSIFICATION")
print("=" * 60)
print("\nUpload your dataset.\nSupported formats: CSV, XLSX, XLS")

try:
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) == 0:
        raise ValueError("No file was uploaded.")
    filename = list(uploaded.keys())[0]
except ImportError:
    raise RuntimeError("This program is designed for Google Colab.")

print("\nUploaded file:", filename)

try:
    if filename.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(uploaded[filename]), sep=None, engine="python")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(uploaded[filename]), sep=None, engine="python", encoding="latin1")
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(uploaded[filename]))
    else:
        raise ValueError("Unsupported file format.")
except Exception as e:
    print("\nERROR READING DATASET:\n", e)
    raise

df = df.dropna(axis=0, how="all")
df = df.dropna(axis=1, how="all")
df.columns = df.columns.astype(str).str.strip()

if df.empty or len(df.columns) < 3:
    raise ValueError("The dataset must contain at least 3 columns.")

print("\nDATASET LOADED SUCCESSFULLY")
print("Number of Rows    :", df.shape[0])
print("Number of Columns :", df.shape[1])
display(df.head())

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_columns) < 2:
    raise ValueError("At least TWO numerical feature columns are required.")

x_feature_dropdown = widgets.Dropdown(options=numeric_columns, value=numeric_columns[0], description="X-axis:", style={"description_width": "initial"}, layout=widgets.Layout(width="400px"))
y_feature_dropdown = widgets.Dropdown(options=numeric_columns, value=numeric_columns[1], description="Y-axis:", style={"description_width": "initial"}, layout=widgets.Layout(width="400px"))

if "Purchased" in df.columns:
    default_target = "Purchased"
else:
    default_target = df.columns[-1]

target_dropdown = widgets.Dropdown(options=list(df.columns), value=default_target, description="Target:", style={"description_width": "initial"}, layout=widgets.Layout(width="400px"))
k_slider = widgets.IntSlider(value=3, min=1, max=15, step=1, description="K:", style={"description_width": "initial"}, layout=widgets.Layout(width="400px"))
run_button = widgets.Button(description="Run KNN", button_style="success", icon="play", layout=widgets.Layout(width="200px", height="40px"))
output_area = widgets.Output()

display(x_feature_dropdown, y_feature_dropdown, target_dropdown, k_slider, run_button, output_area)

def run_knn(button):
    with output_area:
        output_area.clear_output(wait=True)
        x_feature = x_feature_dropdown.value
        y_feature = y_feature_dropdown.value
        target_column = target_dropdown.value
        K = k_slider.value

        if x_feature == y_feature:
            print("ERROR: X-axis and Y-axis features must be different.")
            return

        data = df[[x_feature, y_feature, target_column]].dropna()
        if len(data) < 10:
            print("ERROR: Dataset contains too few valid samples.")
            return

        X = data[[x_feature, y_feature]]
        y = data[target_column]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.astype(str))

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if K > len(X_train_scaled):
            print(f"ERROR: K={K} exceeds training sample count.")
            return

        model = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print("RESULTS")
        print("Total Samples     :", len(data))
        print("Training Samples  :", len(X_train))
        print("Testing Samples   :", len(X_test))
        print("Accuracy          :", f"{accuracy * 100:.2f}%")
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

        X_visual = X_train_scaled
        test_point = X_test_scaled[0]
        actual_class = y_test[0]
        distances = np.sqrt(np.sum((X_visual - test_point) ** 2, axis=1))
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:K]

        fig, ax = plt.subplots(figsize=(9, 6))
        for class_value in np.unique(y_train):
            points = X_visual[y_train == class_value]
            ax.scatter(points[:, 0], points[:, 1], s=70, label=f"Class {label_encoder.classes_[class_value]}")
        ax.scatter(test_point[0], test_point[1], marker="*", s=350, edgecolors="black", linewidth=2, label="Test Point")
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title("KNN Classification")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()

run_button.on_click(run_knn)
