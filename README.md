

**Name: Jonathan zlatin** 😊

---

**Description:**
This repository contains Python code for implementing a Hidden Markov Model (HMM) for part-of-speech tagging using the Viterbi algorithm. The program is designed to analyze text data, predict part-of-speech tags for words in a given sentence, and evaluate the accuracy of the predictions. 💻

---

**Features:**
1. **Data Processing:** The program preprocesses text data from the Brown corpus, dividing it into training and test sets for model training and evaluation. 📊
2. **Model Training:** It builds a Hidden Markov Model by calculating transition probabilities between tags and emission probabilities of words given tags. 🤖
3. **Viterbi Algorithm:** The core of the program is the implementation of the Viterbi algorithm, which computes the most likely sequence of tags for a given sentence. 🧮
4. **Evaluation:** The program evaluates the accuracy of the predicted tags on both known and unknown words in the test set, providing error rates for analysis. 📉
5. **Enhancements:** It offers options for enhancing the model, including Laplace smoothing (add-one) and pseudo-word handling, to improve performance. 🚀

---

**Usage:**
1. **Installation:** Ensure that Python 3.x is installed on your system along with the required libraries listed in `requirements.txt`. You can install them using pip:
   ```
   pip install -r requirements.txt
   ```
2. **Running the Code:** Execute the `main.py` file to run the program. It will preprocess the data, train the HMM model, predict tags for the test set, and evaluate performance.
   ```
   python main.py
   ```


**Acknowledgments:**
- The code is based on concepts and techniques learned from various sources, including textbooks, online tutorials, and academic courses
  primerly thought in the Hebrew University.

---

**Disclaimer:**
This project is for educational and research purposes only. While efforts have been made to ensure the accuracy and reliability of the code, no guarantees are provided. Users are encouraged to review and validate the results before making any decisions based on them. ⚠️
