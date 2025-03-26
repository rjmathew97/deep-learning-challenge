# Module 21 (Deep Learning Challenge)

## **Overview**
The purpose of this analysis is to create a binary classification model using a deep learning neural network that predicts whether charitable organization applications funded by Alphabet Soup will be successful.

This model enables Alphabet Soup to determine which organizations are likely to be successful in securing funding based on various features such as application type, affiliation, use case, and requested funding amount.

* **The Dataset** - The dataset used for this analysis contains information about charitable organization and their grant application like APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, and IS_SUCCESSFUL. The feature columns serve as inputs to train the neural network model, while IS_SUCCESSFUL indicates the binary outcome: 1 as successful and 0 as not successful

* **Stages of the Machine Learning Process** - The stages of the machine learning process are very scripted. If followed in order, they provide you with an accurate assessment of the model's ability to make predictions. The stages of the machine learning process are as follows: 
1. Prepare the data by dropping non-informative columns and encoding categorical variables with one-hot encoding.
2. Separate the dataset into features and the target label (IS_SUCCESSFUL).
3. Split the dataset into training and testing sets (75% / 25%).
4. Build and train a Neural Network model using TensorFlow/Keras with binary crossentropy loss and the Adam optimizer.
5. Evaluate the model's performance using test accuracy and loss metrics.

## **Technologies Used**
- Python 
- Google Colab
- Pandas
- scikit-learn 
    - StandardScaler
	- train_test_split
- TensorFlow / Keras
    - Sequential
	- Dense
	- ModelCheckpoint

deep-learning-challenge/
│   Starter_Code_part_1_and_2.ipynb     # Colab Notebook with preprocessing and initial model
│	Starter_Code_part_3_and_4.ipynb 	# Colab Notebook with model optimization attempts
│  	AlphabetSoupCharity.h5               # Saved HDF5 file of the initial neural network model
│   AlphabetSoupCharity_Optimization.h5  # Saved HDF5 file of the optimized model
│   README.md          					 # Project documentation (this file)

## **Setup Instructions**
1. **Clone the Repository:**
Clone the GitHub repository and navigate into the project directory

2. **Install the Dependencies:**
Pandas, tensorflow, scikit-learn are installed 

3. **Load and Explore the Data :**
Analyze features and check for missing values

4. **Preprocess the Data:** 
One-hot encode categorical columns

Scale numerical features

5. **Train Neural Network Model:**
Fit the model with training data

6. **Evaluate the Model:**
Predict test data and compute accuracy

## **Results**
Machine Learning Model - Neural Network (Sequential Model)
* **Compiling, Training, and Evaluating the Model:** 
	* **Input Layer:** 
		* Number of features after encoding
	* **Hidden Layers:** 
		* 2 hidden layers with 80 and 30 neurons respectively, using ReLU activation
	* **Output Layers:** 
		* 1 neuron with sigmoid activation
	* **Reasoning:**
		* ReLU was chosen for hidden layers to handle non-linearity efficiently, and Sigmoid was used in the output layer for binary classification.
		
* **Final Model Structure:** 
	* **Accuracy:** 73%


## Summary
The **Neural Network Model** achieved **73% accuracy**, performing reasonably well in predicting application success, though it fell slightly short of the 75% target.

However, class imbalance and complex categorical features limited precision on edge cases. Further improvements like hyperparameter tuning or increasing model complexity could enhance performance.

## Recommendation for a Different Model:
Suggested Model: **Random Forest Classifier**
-	It handles categorical data and high-cardinality features well. 
-	Provides feature importance insights
-	Is robust to outliers and noise
-	Requires less feature scaling compared to neural networks

A Random Forest model could potentially improve accuracy and interpretability, making it a good alternative for this classification problem.


