# 🐱🐶 Cat vs Dog Image Classification using Logistic Regression (Sigmoid) & SVM

This project demonstrates **binary image classification** using two classic Machine Learning algorithms — **Logistic Regression (Sigmoid function)** and **Support Vector Machine (SVM)**.  
The goal is to correctly classify images as either **Cat 🐱** or **Dog 🐶** using feature extraction from images.

---

## 🚀 Project Overview

This project was developed and trained on **Kaggle** using the **Cats vs Dogs** dataset.  
It shows how traditional ML algorithms (Logistic Regression and SVM) can achieve high accuracy on image data when combined with feature extraction techniques.

---

## 📂 Dataset

The dataset used is from Kaggle:  
➡️ [Dogs vs Cats Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

**Structure:**
```
/training_set/
    cats/
    dogs/
/test_set/
    cats/
    dogs/
```

- **Total Training Images:** 8,005  
- **Total Test Images:** 2,023  
- **Classes:** Cats, Dogs

---

## 🧠 Models Used

### 1. Logistic Regression (Sigmoid Function)
- A linear classifier that uses the **Sigmoid function** to map probabilities.
- Ideal for binary classification tasks.

### 2. Support Vector Machine (SVM)
- A non-linear classifier using **RBF kernel**.
- Finds the optimal hyperplane to separate classes in higher-dimensional space.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|-----------|
| Logistic Regression | 99.16% | 0.99 | 0.99 | 0.99 |
| SVM (RBF Kernel) | 98.91% | 0.99 | 0.99 | 0.99 |

**Confusion Matrix (Logistic Regression):**
```
[[1003    8]
 [   9 1003]]
```

**Confusion Matrix (SVM):**
```
[[998   13]
 [  9 1003]]
```

---

## ⚙️ How to Use This Code

### Step 1 — Clone the Repository
```bash
git clone https://github.com/<your-username>/cat-vs-dog-classification.git
cd cat-vs-dog-classification
```

### Step 2 — Install Dependencies
Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

Install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 3 — Download the Dataset
Download the **Cats vs Dogs** dataset from Kaggle and place it like this:
```
data/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

### Step 4 — Run the Notebook
Simply open the Jupyter Notebook:
```bash
jupyter notebook cats_dogs_classification.ipynb
```
Or run on Kaggle:  
Upload the notebook and run all cells — it will automatically:
- Load the dataset  
- Extract features  
- Train Logistic Regression  
- Train SVM  
- Display Confusion Matrices and Accuracy

---

## 🧾 Requirements

All libraries used in this project are common in machine learning:

```
tensorflow
scikit-learn
numpy
matplotlib
```

---

## 🧩 Code Workflow

1. **Import and Preprocess Data**
   - Load training and test images using TensorFlow’s `image_dataset_from_directory()`.
   - Resize all images to 64x64.
2. **Feature Extraction**
   - Use a pretrained CNN (like MobileNetV2) to extract 2048-dimensional feature vectors.
3. **Model Training**
   - Logistic Regression with Sigmoid activation.
   - SVM with RBF kernel.
4. **Evaluation**
   - Compute accuracy, precision, recall, and F1-score.
   - Display confusion matrix and performance summary.
5. **Visualization**
   - Plot sample predictions and performance metrics.

---

## 📸 Sample Output

Example Output:
```
Training Logistic Regression ...
Accuracy: 99.15%

Training SVM ...
Accuracy: 98.91%

Confusion Matrices and Classification Reports displayed below.
```

---

## 🧑‍💻 Author

**Hamza Munir**  
🎓 Bachelor of Software Engineering — Superior University, Lahore  
💻 Machine Learning & Web Development Enthusiast  
📧 [itxhamzamunir@gmail.com](mailto:itxhamzamunir@gmail.com)  
🌐 [GitHub Profile](https://github.com/hamzamunir)

---

## ⭐ Acknowledgements

- [Kaggle Dataset — Cats and Dogs](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- TensorFlow for feature extraction
- Scikit-learn for Logistic Regression and SVM implementations

---

## 🏁 Conclusion

This project proves that **classical machine learning algorithms** like Logistic Regression and SVM can still perform impressively on image classification tasks when paired with **deep feature extraction**.  
It’s simple, efficient, and perfect for demonstrating understanding of both **traditional ML** and **modern feature engineering**.
