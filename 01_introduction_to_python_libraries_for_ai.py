import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("INTRODUCTION TO PYTHON LIBRARIES FOR AI")

# NUMPY
print("\n-----------Numpy----------")
numbers = np.array([10, 20, 30, 40, 50])
print("Array:", numbers)
print("Sum:", np.sum(numbers))
print("Mean:", np.mean(numbers))
print("Maximum:", np.max(numbers))

# PANDAS
print("\n----Pandas----")
student = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Marks": [85, 90, 78]
}
df = pd.DataFrame(student)
print("\nStudent Data")
print(df)

# MATPLOTLIB
print("\n----Matplotlib----")
print("Displaying Bar Chart")
plt.figure(figsize=(6, 4))
plt.bar(df["Name"], df["Marks"])
plt.title("Student Marks")
plt.xlabel("Student Name")
plt.ylabel("Marks")
plt.show()

# SCIKIT-LEARN
print("\n----Scikit-Learn-----")
encoder = LabelEncoder()
df["Student ID"] = encoder.fit_transform(df["Name"])
print("After Encoding Names into Numbers")
print(df)

# TENSORFLOW
print("\n-----TensorFlow----")
tensor1 = tf.constant([1, 2, 3])
tensor2 = tf.constant([4, 5, 6])
print("Tensor 1:", tensor1.numpy())
print("Tensor 2:", tensor2.numpy())
result = tf.add(tensor1, tensor2)
print("Tensor Addition:", result.numpy())

# OPENCV
print("\n----OpenCV----")
image = np.zeros((250, 250, 3), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), 3)
cv2.circle(image, (125, 125), 40, (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("OpenCV Shapes")
plt.axis("off")
plt.show()
