# Coffee Quality Prediction

This project tries to predict the cupping score of a coffee sample employing machine learning models.

The total cupping score measures a coffee quality and it is judged on a 100 point scale. Any coffee scoring above 80 earns the title of "Specialty Coffee". The cupping score is obtained by suming 10 sensory sub-scores: Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean.Cup, Sweetness and Cupper Points.  

Nowadays, knowing the actual cupping score of your coffee is vastly accesible and cheap. However, true access is still unequal, specially for small farmers located in remote regions that rely on local cooperatives or exporters to grade their coffee. Having an approximation of the true cupping score would allow them to have control over their own product and to define a fair price.

Therefore, in this ML project, we will try to predict the cupping score considering only non-sensory features.

## Dataset

The dataset was extracted from the Coffee Quality Institute in 2018, and can be found [here](https://jldbc.github.io/coffee-quality-database)



1. An exploratory data analysis, the cleaning and selection of training features can be found in [coffe_data_analysis](https://nayualv.github.io/Arabica_coffee_quality/notebooks/coffee_data_analysis.ipynb)

2. Implementations of various Machine learning models to predict the coffee cup scores, along with performance discussions can be found in [coffee_data_training](https://nayualv.github.io/Arabica_coffee_quality/notebooks/coffee_data_train.ipynb)


## Setup

```bash
pip install -r requirements.txt
```