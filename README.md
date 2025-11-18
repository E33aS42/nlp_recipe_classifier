# Recipes classifier using NLP

### Presentation

This was a personal project whose objective was to build a model capable of predicting the country of origin of a given recipe. This is an NLP classification problem, as each recipe is described by its list of ingredients.

The data sources are open data from a competition on the Kaggle website (https://www.kaggle.com/c/whats-cooking/data).

This project has been done in three parts:

1. Cleaning and tokenization of the raw database.  
→ Convert everything to lowercase  
→ Remove:  
    • accents  
    • numbers  
    • special characters and punctuation (&, %, !, /, ‘, ® …)  
    • “stopwords” (at, with, the…)  
    • additional unnecessary words (large, fine, too…)  
    • spelling mistakes  
    • plurals  
→ We also removed terms with a frequency lower than 3 (outliers) to better generalize our model.

Regarding the variables (ingredients), it was chosen to separate each word of each ingredient (tokenization), since the algorithms prove to perform better with this method, rather than concatenating all the words of an ingredient into a single token. This avoided having multiple different variable names referring to the same ingredient. However, we ignored the correlation effect between the different words that make up an ingredient.

2. Binarization of the data.

Each country of origin was represented by a number from 1 to 20.
The ingredient lists were vectorized into binary vectors (0s and 1s) and grouped into a single matrix, where each row corresponds to a recipe and each column to a unique ingredient.

A second representation, known as a unigram (TF-IDF), was also implemented by adjusting the weight of the words based on their frequency within each recipe and across all recipes.


3. Training and comparison of different classification models: Random Forest, XGBoost, Naive Bayes, and Logistic Regression.

We first split our data into a training set and a test set, with the latter set aside.

We defined a parameter grid search on which we evaluated the model’s performance using the training set.
We looped over all possible combinations of the parameters of interest and evaluated our classifier for each combination using cross-validation.

We then selected the parameter combination that yielded the best performance for each type of algorithm, and then applied it to the test set for the final evaluation.



   
