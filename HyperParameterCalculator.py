import math
from Train import *

def predict_target_values(test_X, model):
    predict_Y=[]
    for X in test_X:
        node=model
        while node.left:
            if X[node.feature_index]<node.threshold :
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


def hypermaterKnowing(train_X, train_Y, validation_split_percent):
    length_training_Data_X=math.floor(((float(100-validation_split_percent))/100)*len(train_X))
    training_Data_X=train_X[0:length_training_Data_X]
    training_Data_Y=train_Y[0:length_training_Data_X]
    max_depth=[i for i in range(1,len(train_X[0])+4)]
    min_size=[i for i in range(len(list(set(train_Y)))+2)]
    for i in max_depth:
        for j in min_size:
            root=construct_tree(training_Data_X,training_Data_Y,i,j,0)
            pred_Y = predict_target_values(train_X ,root)
            write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")
            validate("train_X_de.csv", actual_test_Y_file_path="train_Y_de.csv") 
            print(" this is accuracy for above max_depth---",i)
            print(" this is accuracy for above min_size---",j)
            

X,Y=Import_data()
X=data_processing(X)
hypermaterKnowing(X,Y,30)