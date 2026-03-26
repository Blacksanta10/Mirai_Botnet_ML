# Purpose is to execute different parts of workflow 

from sklearn.model_selection import train_test_split
from src.processing import final_load_data
from src.train import train_knn
from src.testing import evaluate

def main():
    ### Load data ###
    X, y = final_load_data()

    ### Split data ###
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    ### Train model ###
    model = train_knn(X_train, y_train, n_neighbors=5)

    ### Evaluate ###
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()