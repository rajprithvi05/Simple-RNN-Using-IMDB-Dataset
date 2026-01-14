# 🎬 Sentiment Analysis on IMDB Reviews using Simple RNN

A deep learning project that performs **binary sentiment classification** (Positive / Negative) on movie reviews using a **Simple Recurrent Neural Network (RNN)** trained on the **IMDB dataset**.

This project is designed to understand the fundamentals of **RNNs in Natural Language Processing (NLP)** using TensorFlow and Keras.

---

## 🚀 Highlights

- Uses IMDB movie review dataset
- Implements a Simple RNN for sequence learning
- Early stopping to prevent overfitting
- Visualizes training & validation performance
- Saves and reloads trained model
- Predicts sentiment for unseen reviews

---

## 📌 Project Overview

- **Task:** Sentiment Analysis (Binary Classification)
- **Input:** Tokenized movie reviews
- **Output:** Positive or Negative sentiment
- **Model Type:** Simple Recurrent Neural Network
- **Framework:** TensorFlow (Keras API)

---

## 📂 Project Structure

imdb-simple-rnn/
│
├── SimpleRNN.py # Model training, evaluation & prediction
├── simple_rnn_imdb.h5 # Saved trained model
├── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Tech Stack

| Category | Tools |
|-------|------|
| Programming Language | Python 3.x |
| Deep Learning | TensorFlow, Keras |
| Dataset | IMDB (keras.datasets) |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Model | Simple RNN |

---

## 📊 Dataset Details

- **Dataset:** IMDB Movie Reviews
- **Total Samples:** 50,000
- **Vocabulary Size:** 1,000 most frequent words
- **Train-Test Split:** 80% / 20%
- **Max Review Length:** 200 tokens (after padding)

---

## 🧠 Model Architecture

Embedding Layer (input_dim=1000, output_dim=128)
↓
SimpleRNN Layer (128 units, tanh activation)
↓
Dense Layer (1 unit, sigmoid activation)

yaml
Copy code

### Compilation Details
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy

---

## 🚀 Training Configuration

- **Epochs:** Up to 20
- **Batch Size:** 32
- **Validation Split:** 20%
- **Callbacks:** EarlyStopping  
  - Monitors validation loss  
  - Patience = 5  
  - Restores best model weights  

---

## 📈 Performance Visualization

The script plots:
- Training vs Validation Accuracy
- Training vs Validation Loss

These plots help analyze:
- Overfitting
- Underfitting
- Model convergence behavior

---

## 🧪 Model Evaluation

After training, the model is evaluated on the test dataset:

Test Loss: <value>
Test Accuracy: <value>

csharp
Copy code

The trained model is saved as:

simple_rnn_imdb.h5

yaml
Copy code

---

## 🔍 Sample Prediction

The model predicts sentiment for a single padded review:

```python
prediction = model.predict(sample_review.reshape(1, -1))
sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
Output Example:

mathematica
Copy code
Predicted Sentiment: Positive
Actual Label: Positive
▶️ How to Run the Project
1️⃣ Install Dependencies
bash
Copy code
pip install tensorflow numpy matplotlib
2️⃣ Run the Training Script
bash
Copy code
python SimpleRNN.py
3️⃣ Model Output
Training begins automatically

Accuracy/Loss graphs are displayed

Model saved as simple_rnn_imdb.h5

🧩 Future Improvements
Replace SimpleRNN with LSTM or GRU

Increase vocabulary size

Decode tokenized reviews to readable text

Add confusion matrix

Deploy using Streamlit or Flask

Convert into REST API

🎓 Learning Outcomes
Understanding sequence modeling with RNNs

Text preprocessing and padding

Binary classification using neural networks

Model saving and reloading

Performance visualization

📜 License
This project is intended for educational and learning purposes.

✨ A beginner-friendly NLP project to grasp the essence of Recurrent Neural Networks.

yaml
Copy code

---

If you want, I can also:
- Convert this into **resume-ready project description**
- Add **GitHub badges** (Python, TensorFlow, License)
- Create a **Streamlit app** for live sentiment prediction
- Refactor code into **modular files**



👤 Author

Prithviraj Chouhan
Python | SQL | AI & ML Enthusiast

⭐ Acknowledgement

Thanks to open-source datasets and libraries that made this project possible.


---
📞Connect with me:


📧Email: rajprithvi05oct@gmail.com
🔗Linkedin: https://www.linkedin.com/in/prithvi25


Just tell me the next move 🚀










