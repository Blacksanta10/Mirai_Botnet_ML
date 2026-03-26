""" Step 3: Purpose is to create model predictions,
then to show different metrics based on predictions 
""" 

from sklearn.metrics import classification_report, accuracy_score

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassififcation Report:")
    print(classification_report(y_test, predictions))   #text report
