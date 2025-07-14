# Sentiment Analysis with Deep Learning - Explanation

## Data Read

### Process Description

This section focuses on loading the dataset for sentiment analysis. The process involves:

1. Importing essential data handling libraries: `numpy` for numerical operations and `pandas` for data manipulation
2. Reading the training data from 'twitter_training.csv' into a DataFrame named `df`
3. Reading the validation data from 'twitter_validation.csv' into a DataFrame named `df_val`
4. Displaying the first 10 rows of both the training and validation datasets to understand their structure
5. Checking for missing values in both datasets, which reveals that the training dataset has 686 null values in the review column, while the validation dataset has no null values

## Data Preprocessing

### Process Description

This section focuses on cleaning and preparing the data for modeling. The process includes:

1. Removing unnecessary columns from the training dataset (dropping the first column containing IDs)
2. Renaming the columns to more descriptive names: "Keyword", "Sentiment", and "Content"
3. Removing rows with missing values to ensure data quality
4. Examining the dataset information to confirm the changes and understand the data structure (73,995 entries with 3 columns)
5. Similarly renaming and organizing the validation dataset for consistency
6. Analyzing the distribution of sentiment classes in both datasets to understand the class balance
7. Creating separate content and entity length analysis to understand the text characteristics
8. Visualizing the distribution of text lengths to inform preprocessing decisions

## Train Test Split

### Process Description

This section divides the dataset into training and testing sets for model development and evaluation. The process involves:

1. Importing the necessary functions from scikit-learn for dataset splitting and label encoding
2. Creating separate feature (X) and target (y) variables from the training dataset
3. Splitting the data into training (80%) and testing (20%) sets using stratified sampling to maintain class distribution
4. Further splitting the test set to create a validation set for model tuning
5. Encoding the sentiment labels (converting categorical labels to numerical values) using LabelEncoder
6. Preparing separate versions of the data for both the content text and entity (keyword) information
7. Setting up the data structure for a multi-input model that will use both text content and entity information

## Natural Text to Token

### Process Description

This section transforms the raw text data into numerical tokens that can be fed into a neural network. The process includes:

1. Setting key parameters for the tokenization process:
   - Maximum vocabulary size (10,000 words)
   - Maximum sequence length for content (75 tokens)
   - Maximum length for entity sequences (1 token)
   - Embedding dimension (100)

2. Creating two separate tokenizers:
   - One for the main text content, which learns a vocabulary from the training data
   - Another for the keyword entities

3. Converting text to sequences of integers based on the learned vocabularies

4. Padding sequences to ensure uniform length:
   - Content sequences are padded to 75 tokens
   - Entity sequences are set to exactly 1 token

5. Preparing the data in the proper format for a dual-input model that will process both content and entity information separately

This tokenization approach allows the model to process both the main text and the entity information as separate but complementary inputs.

## Train

### Process Description

This section builds and trains a dual-input deep learning model for sentiment analysis. The process involves:

1. Creating a complex model architecture with two separate input branches:
   - Content branch: Processes the main text through embedding, LSTM, and dense layers
   - Entity branch: Processes the keyword through embedding and dense layers

2. Combining the two branches using concatenation before the final classification layer

3. Configuring the model with:
   - Categorical cross-entropy loss function (appropriate for multi-class classification)
   - Adam optimizer with a specific learning rate
   - Accuracy as the evaluation metric

4. Training the model using both content and entity inputs:
   - Using mini-batches of 128 samples
   - Training for multiple epochs (iterations over the data)
   - Using early stopping to prevent overfitting
   - Validating performance on the validation set during training

5. The model's progress is monitored throughout training, showing gradual improvement in classification accuracy

This dual-input approach leverages both the context from the full text and the specific entity information to achieve better sentiment classification performance.

## Evaluation

### Process Description

This section assesses the performance of the trained sentiment analysis model. The evaluation process includes:

1. Generating predictions on the test dataset using the trained model

2. Converting the probability outputs to class predictions by taking the argmax

3. Calculating key performance metrics:
   - Overall accuracy on the test set
   - Precision, recall, and F1-score for each sentiment class
   - Confusion matrix to visualize prediction patterns

4. Creating visualizations of the model performance:
   - Confusion matrix heatmap showing the distribution of predictions vs. actual labels
   - Classification report detailing per-class performance

5. Analyzing where the model performs well and where it struggles:
   - Identifying which sentiment classes are easier or harder to predict
   - Understanding potential patterns in misclassifications

The evaluation shows the model achieves approximately 93% accuracy on the test set, demonstrating strong performance in distinguishing between different sentiment classes in the Twitter data.
