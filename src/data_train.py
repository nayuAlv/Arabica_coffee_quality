import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt


def cup_score_class(score):
    if score < 80:   
        return 1
    else: 
        return 0
    
def choose_threshold(estimator, X_train, y_train, beta= 0.5, random_state=1):
    """
    Pick the best probability threshold that maximises F-beta, using out-of-fold predictions on the training 
    set only. For beta < 1 weights, the metric weights precision more than recall.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probs = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba")[:,1]
    thresholds = np.linspace(0.05, 0.95, 181)
    fbetas = [fbeta_score(y_train, (oof_probs >= t).astype(int), beta= beta, pos_label=1, zero_division=0) for t in thresholds ]

    best_i = int(np.argmax(fbetas)) 
    return thresholds[best_i], fbetas[best_i]

def train_logistic_regression(preprocessor, X_train, y_train, scoring='average_precision', random_state=1, param_grid=None):
    pipe = Pipeline([
        ("pre", preprocessor),
        ("lr", LogisticRegression(class_weight='balanced',
                                   random_state=random_state, max_iter=1000)),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring=scoring, 
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid

def train_random_forest(preprocessor, X_train,y_train, scoring='average_precision', random_state=1,class_weight='balanced', param_grid=None):
    pipe = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestClassifier(class_weight=class_weight,
                                       random_state=random_state, n_jobs=-1)),
    ])
    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scoring,  
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def train_xgb_model(preprocessor, X_train, y_train, scoring='average_precision', 
                    random_state=1, param_grid=None):
    
    optimal_scale = np.sum(y_train == 0) / np.sum(y_train == 1)      
    
    pipe = Pipeline([
    ("pre", preprocessor),
    ("xgb", XGBClassifier(
    learning_rate=0.1,
    scale_pos_weight=optimal_scale,
    random_state=random_state,
    n_jobs=-1)),
    ])

    grid_search_xgb = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
    scoring=scoring,
    n_jobs=-1
    )

    grid_search_xgb.fit(X_train, y_train)

    return grid_search_xgb

def plot_precision_recall_vs_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    precision, recall = precision[:-1], recall[:-1]
    plt.figure(figsize=(5, 3))
    plt.plot(thresholds, precision, label="Precision")
    plt.plot(thresholds, recall, label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

def evaluate_at_seed(random_state, X, y, preprocessor, beta= 0.5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=random_state, stratify=y
    )

    lr_grid = train_logistic_regression(
        preprocessor, X_train, y_train, random_state=random_state,
        param_grid={'lr__C': [0.001, 0.01,0.05, 0.1, 1]},
    )
    rf_grid = train_random_forest(
        preprocessor, X_train, y_train, random_state=random_state, class_weight='balanced',
        param_grid={
            "rf__n_estimators": [100, 200, 300],
            "rf__max_features": ['sqrt', 0.3, 0.5],
            "rf__min_samples_leaf": [2, 3, 5, 7],
            "rf__max_depth": [None, 2, 3],
        },
    )

    xgb_grid = train_xgb_model(
        preprocessor, X_train, y_train,random_state=random_state, 
        param_grid = {
    'xgb__n_estimators': [30, 50, 70, 90],
    'xgb__max_depth': [2, 3, 4, 5, 6],
    'xgb__min_child_weight': [7, 9, 11, 13, 15]
    },
    )
    
    rows = []
    for name, grid in [("Logistic Regression", lr_grid), ("Random Forest", rf_grid), ("XGBoost", xgb_grid)]:
        model = grid.best_estimator_
        t, _ = choose_threshold(model, X_train, y_train, beta=beta, random_state=random_state)
        y_pred = (model.predict_proba(X_test)[:, 1] >= t).astype(int)
        rows.append({
            "seed": random_state,
            "model": name,
            "threshold": t,
            "precision_1": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall_1": recall_score(y_test, y_pred, pos_label=1),
            "fbeta_1": fbeta_score(y_test, y_pred, beta=beta, pos_label=1, zero_division=0),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
        })
    return rows