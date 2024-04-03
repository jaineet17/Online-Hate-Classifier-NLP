# Online Hate Classifier
## Project Overview
The Online Hate Classifier is a machine learning project aimed at identifying hate speech, offensive language, and neutral content in tweets. Utilizing a variety of machine learning models and natural language processing (NLP) techniques, this project seeks to automate the detection of online hate speech effectively.

## Installation
This project requires Python 3.x and the following Python libraries installed:

Pandas <br>
NumPy <br>
Matplotlib <br>
Seaborn <br>
NLTK <br>
Scikit-learn <br>
TensorFlow <br>
Keras <br>
XGBoost <br>
vaderSentiment <br> <br>
You can install these libraries using pip:

```pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow keras xgboost vaderSentiment``` <br> <br>
Additionally, you will need to download some NLTK datasets:
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
``` 
## Dataset
The dataset contains tweets labeled according to the presence of hate speech, offensive language, or neither. It includes columns for tweet content, counts of different classifications, and a final class label. The dataset is pre-processed to tokenize and vectorize the tweets using TF-IDF, and features are generated to capture the syntactic structure of the tweets and other textual properties.

## How to Run
Clone this repository to your local machine.
Ensure you have all the required libraries installed.
Load your dataset in a CSV format similar to the provided structure.
Run the Jupyter Notebook to preprocess the data, train models, and evaluate their performance.
```
jupyter notebook Nlp_IA.ipynb
```
## Methodology
### Preprocessing: 
Tokenization, removing stopwords, and vectorization using TF-IDF.
### Feature Engineering: 
Generating features like sentiment scores, part-of-speech tags, and syntactic structure.
### Model Training: 
Several models are trained, including Logistic Regression, Decision Trees, Artificial Neural Networks (ANN), Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).
### Evaluation: 
Models are evaluated based on their accuracy, precision, recall, and F1 score.
## Models and Performance
The project explores various models, adjusting parameters and utilizing techniques like cross-validation and grid search to find optimal settings. Performance metrics are provided for each model, demonstrating their effectiveness in classifying tweets.

## Contributing
Contributions to this project are welcome! Please fork the repository and open a pull request with your proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
