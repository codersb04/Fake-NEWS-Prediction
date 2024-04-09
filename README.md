# Fake-NEWS-Prediction
## Task:
Build a Machine Learning model To predict whether the given news is real or fake.</br>
The problem is based on <b>Supervised learning</b> as we are provided labelled data. Also, it comes under the binary classification problem as the outcome can only be either 0(real news) or 1(fake news). </br>
We will be using <b>Logistic Regression</b> to solve to achieve the result.

## Dataset:
The dataset contains around 20800 data with 5 different features that are: </br></br>
**id**: unique id for a news article</br>
**title**: the title of a news article</br>
**author**: author of the news article</br>
**text**: the text of the article; could be incomplete</br>
**label**: a label that marks the article as potentially unreliable</br>
  1: unreliable</br>
  0: reliable</br></br>

  Source: Fake News, Build a system to identify unreliable news articles, Kaggle,https://www.kaggle.com/competitions/fake-news/data

  ## Steps involved:
### **Import Dependencies**:</br>
Libraries need are: numpy, pandas,re(regular expression), nltk(Natural Language Tool Kit), PorterStemmer, sklearn, TfidfVectorizer LogisticRegression and metrics.</br>
Other than these we need to download the list of stopwords from NLTK libraries.</br></br>
### **Data Collection and Data Processing**:</br>
The first step to this is to import the dataset to the pandas data frame using the read_csv function.</br> 
Since the dataset is all textual based and computer can only understand numerical data so we there lot of data processing steps involved in this that are:</br>
* Check for the missing value</br>
* Handling the missing value: replace the null value with an empty string</br>
* Combine the Author and Title in one column as 'Content'.</br>
* Stemming: It is a process of reducing the word to the root word. In this step, we will also remove all the non-alphabetical items from the text by using regular expressions. Also, We will remove the stopwords from the text during this.</br>
* Separate the data and the label: For the data part we will only consider the content(Author and Title Combined) rest will be ignored and for the label, we will take the last column which tells whether the news is real or fake.</br>
* Converting text to numerical Values: This is the final step of data processing where we will convert all the text to numeric values based on their importance using TfidfVectorizer.</br></br>
### **Splitting the data into training and Test sets**:</br>
This will be done using the function test_train_split from sklearn. mode_selection. We will take 20% data for testing and the rest 80% for training.</br></br>
### **Training the model**:</br>
Used Logistic Regression for building the model</br>
Sigmoid function:</br>
Y = 1/1+e**-Z,</br>
  Z= w*X +b,</br>
  X: Input Features,</br>
  Y: Prediction Probability,</br>
  w: weights(depends upon the importance of each column or feature),</br>
  b: bias</br></br>
### **Model Evaluation**:</br>
Model evaluation is done by feeding the data to the model and predicting the respective outcome and afterwards, checking the accuracy score by comparing the prediction with the actual result. </br>
We are performing the evaluation on both the trained data and test data. The accuracy score for them is:</br>
Trained data: 0.9753605769230769</br>
Test data: 0.9634615384615385</br></br>
### **Build a Predictive System**:</br>
This step involves taking random data from the dataset and feeding it to build a model and predict the result.</br></br></br></br></br></br>




Reference: Project 4. Fake News Prediction using Machine Learning with Python | Machine Learning Projects, Siddhardhan, https://www.youtube.com/watch?v=nacLBdyG6jE&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=6

  
  
  

  
