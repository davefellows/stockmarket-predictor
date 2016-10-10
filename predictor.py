
import algorithms
from trainingData import trainingData



def performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel):
    """
    performs classification on daily returns using several algorithms (method).
    method --> string algorithm
    parameters --> list of parameters passed to the classifier (if any)
    fout --> string with name of stock to be predicted
    savemodel --> boolean. If TRUE saves the model to pickle file
    """
   
    if method == 'RF':   
        return performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
        
    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'SVM':   
        return performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'ADA':
        return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'GTB': 
        return performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'QDA': 
        return performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)


if __name__ == '__main__':
    # run training set through various predictive models and evaluate best perf
    
    trainingData.load('SP500', '2014', '2016', 2)
    X_train, y_train, X_test, y_test = trainingData.returnDataForClassification('2015')

    accuracy = performClassification(X_train, y_train, X_test, y_test , 'SVM')
    
    print('SVM accuracy = ' + accuracy)