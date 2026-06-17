# Coffee Quality Prediction

This project tries to predict if a coffee sample belongs to specialty class or below-specialty class employing binary classification models.

The total cupping score measures a coffee quality and it is judged on a 100 point scale. Any coffee scoring >= 80 earns the title of "Specialty Coffee". The cupping score is obtained by summing 10 sensory sub-scores: Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean.Cup, Sweetness and Cupper Points. Nowadays, knowing the actual cupping score of your coffee is vastly accesible and cheap. However, true access is still unequal, especially for small farmers located in remote regions that some rely on local cooperatives or exporters to grade their coffee. Having an approximation of the true cupping score would allow them to have control over their own product and to define a fair price.

Therefore, in this ML project, I will try to predict the cupping score considering only non-sensory features. If you want to know the end of the story, I find that a Random Forest classifier worked the best, achivieng the highest precision for the below-specialty class, and the highest overall macro F1 score. However, none of the models surpassed 40% precision on below-specialty coffee, reflecting our class imbalance and the weak correlation between non-sensory features with the total cupping score.

## Dataset

The dataset was extracted from the Coffee Quality Institute in 2018, and can be found [here](https://jldbc.github.io/coffee-quality-database)

1. An exploratory data analysis, the cleaning and selection of training features can be found in the notebook coffe_data_analysis

2. Implementations of various Machine learning models to classify coffee as specialty or below-specialty, along with a detailed evaluation of their performance can be found in the notebook coffee_data_training


## Setup

```bash
pip install -r requirements.txt
```