import numpy as np
import csv
import sys
import pickle
from validate import validate
from Train import Node

mean_null_list=[[0.5093295 , 0.72749334, 0.13532675, 0.85628456]]
All_mini_maxi_mean_normalize_value=[[-1.20968, 1.91956, 0.5093295043731778], [-0.8058, 4.5333, 0.7274933430515063], [-1.55609, 0.90738,        0.13532675413022352], [-2.75781, 3.33795, 0.8562845578231293]]

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model

def replace_null_values_with_mean(X):
    mean_of_nan=mean_null_list[0]
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize(X, column_indices):
    mini,maxi,mean=All_mini_maxi_mean_normalize_value[column_indices]
    X[:,column_indices]=(X[:,column_indices]-mean)/(maxi-mini)
    return X

def data_processing(class_X) :
    X=replace_null_values_with_mean(class_X)
    for i in range(class_X.shape[1]):
        X=mean_normalize(X,i)
    
    return X

def predict_target_values(test_X, model):
    predict_Y=[]
    for i in range(len(test_X)):
        node=model
        while node.left:
            if test_X[i][node.feature_index]<=node.threshold :
                node=node.left
            else :
                node=node.right
        predict_Y.append(node.predicted_class) 
    return np.array(predict_Y)
    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    test_X=data_processing(test_X)
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 
