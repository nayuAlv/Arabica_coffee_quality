import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier


# Function to define the threshold between Below Specialty and Specialty
def cup_score_class(score):
    if score < 80:   
        return 1
    else: 
        return 0
    
def threshold_tuning(model, X_train, y_train, random_state=1):

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_train_probs = cross_val_predict(model, X_train, y_train, cv=cv_strategy, method="predict_proba")[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_probs)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    optimal_threshold = thresholds[np.argmax(f1_scores)]

    return optimal_threshold




def train_logistic_regression(X_train, y_train, scoring='f1', random_state=1, param_grid={}):
    
    grid = GridSearchCV(
        estimator=LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring=scoring, 
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    return best_model, grid

def train_random_forest(X_train,y_train, scoring='f1', random_state=1,class_weight='balanced', param_grid={}):
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
        param_grid=param_grid,
        scoring=scoring,  
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model, grid_search