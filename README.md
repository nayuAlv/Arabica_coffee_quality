# Coffee Quality Prediction

This project employs binary classification models to predict wether a coffee sample belongs to specialty or below-specialty class using only selected non-sensory features.

1. An exploratory data analysis, cleaning and selection of training features can be found in coffee_data_analysis.ipynb.

2. Implementations of three machine learning models to classify coffee as specialty or below-specialty, along with a detailed evaluation and comparison of their performance, can be found in coffee_data_train.ipynb.


Coffee quality is traditionally judged by certified professionals on a 100-point scale known as the cupping score, obtained by summing 10 sensory sub-scores: Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean.Cup, Sweetness and Cupper Points. Any coffee scoring >= 80 earns the title of "Specialty Coffee". 

The [coffee dataset](https://github.com/jldbc/coffee-quality-database/tree/master/data) contains also non-sensory features such as altitude, moisture, color, and processing type. I will use only a selection of these attributes, which do not require a professional cupping, to train the models.

Since this is a classification problem, the model can be optimized by tuning the decision threshold to balance precision and recall depending on the use case. 

For example, considering an scenario where the main users are small farmers who rely on local cooperatives or exporters to grade their coffee. Providing them with an automated approximation of their coffee class using data they can measure themselves would give them more control over their product and help them negotiate a fair price. 

For that case, a model that prioritizes high precision on the below-specialty class would be ideal, labeling coffee as below-specialty only when it is very confident. I find **Random Forest** to be the most robust and reliable choice compared to Logistic regression and XGBoost. However, since the dataset is skewed (containing only 14% below-specialty coffee samples) and the non-sensory features correlate weakly with the cupping score, all models encounter noticeable noise. Having more below-specialty data would definitely improve the model's  performance.

## Dataset

The dataset was extracted from the Coffee Quality Institute in 2018, and can be found [here](https://github.com/jldbc/coffee-quality-database/tree/master/data)


## Setup

```bash
pip install -r requirements.txt
```