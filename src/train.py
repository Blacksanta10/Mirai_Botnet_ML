# Step 2: Purpose is to conduct model training
from sklearn.neighbors import KNeighborsClassifier


# Example for Nearest Neighbors Classification method
def train_knn(X_train, y_train, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)     #needs to be fit before predicting
    return model

