Data Loading & Preprocessing: Loads the Breast Cancer dataset and selects the first two features for easy 2D visualization. It then splits the data and scales it using StandardScaler.

Model Training: Trains two SVM models — one with a linear kernel and one with an RBF (non-linear) kernel.

Visualization: Plots the decision boundaries of both SVM models to show how they separate the two classes in 2D space.

Hyperparameter Tuning: Uses GridSearchCV to find the best combination of C and gamma for the RBF kernel using cross-validation.

Model Evaluation: Prints the best parameters and the average accuracy across 5-fold cross-validation.

This code demonstrates both linear and non-linear classification using SVMs and how to tune and evaluate them effectively.
