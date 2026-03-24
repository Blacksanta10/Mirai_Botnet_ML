# Step 2: Purpose is to conduct model training
from sklearn.neighbors import KNeighborsClassifier


# Simple function for Nearest Neighbors Classification method
def train_kNN(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

