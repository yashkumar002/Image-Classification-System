# Image-Classification-System


# 🧠 Image Classification System

A powerful and efficient Image Classification System built using Python and Machine Learning libraries. This project demonstrates how to classify images into various categories using a convolutional neural network (CNN) and relevant data preprocessing, training, and evaluation techniques.


## 📽️ Demo Video

Watch a full walkthrough of the Image Classification System, including dataset loading, model training, and live predictions:

[![Watch on YouTube](https://img.youtube.com/vi/abc123XYZ/0.jpg)](https://www.youtube.com/watch?v=abc123XYZ)

📌 *Click the thumbnail or link above to watch the demo on YouTube.*


## 📌 Project Overview

The **Image Classification System** is designed to recognize and categorize images from a dataset into predefined classes. It leverages deep learning frameworks such as TensorFlow/Keras and uses standard datasets or custom datasets (like CIFAR-10, MNIST, or a user-provided image set) to train and validate the model.

## 🎯 Key Features

- 📂 Support for custom and standard image datasets
- 🧠 Deep learning model using Convolutional Neural Networks (CNN)
- 📊 Accuracy tracking with graphs for training and validation
- 🖼️ Real-time image prediction support
- 📝 Confusion matrix and classification report for evaluation
- 💾 Model saving and loading for future inference
- 📸 Jupyter Notebook Interface for ease of use

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- OpenCV (optional for image input)

## 🚀 Getting Started

### 🔧 Installation

Clone the repository and install required packages:

```bash
git clone [https://github.com/yashkumar002/image-classification-system.git](https://github.com/yashkumar002/Image-Classification-System)
cd image-classification-system
pip install -r requirements.txt
````

### 🧪 Dataset

You can use any image classification dataset. To use a custom dataset, ensure it's structured like:

```
/dataset/
    /test/
        img1.jpg
        img2.jpg
        ------
    /validation/
        img1.jpg
        img2.jpg
        ------
    /train/
        img1.jpg
        img2.jpg
        ------
```

### 🧮 Training the Model

Run the Jupyter Notebook:

```bash
jupyter notebook Image_Class_Model-checkpoint.ipynb
```

Follow the steps in the notebook to preprocess data, train the model, and evaluate the results.

### 🔍 Predict on New Images

After training, load your model and use:

```python
from keras.models import load_model
model = load_model('image_classifier_model.h5')

# Predict
img = load_and_preprocess('test_image.jpg')
prediction = model.predict(img)
```

## 📈 Results

* Achieved up to **99% accuracy** on validation dataset
* Visualized training vs. validation loss and accuracy
* Confusion matrix and detailed classification report generated

![Accuracy & Loss Graph](https://github.com/yashkumar002/Image-Classification-System/blob/main/Graph%20for%20%20Accuracy%20and%20loss.png)


## 🧪 Sample Use Cases

* Classify hand-written digits (MNIST)
* Identify animals in images (Cats vs Dogs)
* Recognize medical scans (X-ray, MRI)
* Real-time classification using webcam (with OpenCV)

## 📂 Project Structure

```
📁 image-classification-system/
├── 📁 Project Code/
      ├── 📁 Fruits_Vegetables/
            ├── 📁 test/
                  ├── 📓image_1
                  ├── 📓image_2
                  ------
            ├── 📁 train/
                  ├── 📓image_1
                  ├── 📓image_2
                  ------
            ├── 📁 validation/
                  ├── 📓image_1
                  ├── 📓image_2
                  ------
      ├── 📁 ipynb_checkpoints/
            ├── 📓 Image_Class_Model-checkpoint.ipynb
      ├── 📄 image_classify.keras
├── 📁 Otherfiles/
├── 📁 outputs/
└── 📄 README.md
```

## 🤝 Contributors

* **Yash Banjare** – [LinkedIn](https://www.linkedin.com/in/yash-banjare-199b76264) | [GitHub](https://github.com/yashkumar002)
* **Kishan Kanha** – [LinkedIn](https://www.linkedin.com/in/kishan-kanha-patel-1206a0338) | [GitHub](https://github.com/kanhapatel07)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

🔗 *For any queries or collaborations, feel free to reach out via email or LinkedIn!*

```

---
banjareyash04@gmail.com
```
