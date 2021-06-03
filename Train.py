import csv
import pickle
import numpy as np

mean_null_list=[]
All_mini_maxi_mean_normalize_value=[]

def Import_data():
    X=np.genfromtxt("train_X_de.csv",delimiter=',',dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_de.csv",delimiter=',',dtype=int)
    return X,Y

def replace_null_values_with_mean(X):
    mean_of_nan=np.nanmean(X,axis=0)
    mean_null_list.append(mean_of_nan)
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize(X, column_indices):
    mini=np.min(X[:,column_indices],axis=0)
    maxi=np.max(X[:,column_indices],axis=0)
    mean=np.mean(X[:,column_indices],axis=0)
    All_mini_maxi_mean_normalize_value.append([mini,maxi,mean])
    X[:,column_indices]=(X[:,column_indices]-mean)/(maxi-mini)
    return X

def data_processing(class_X) :
    X=replace_null_values_with_mean(class_X)
    for i in range(class_X.shape[1]):
        X=mean_normalize(X,i)
    
    return X

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None
        

def calculate_gini_index(Y_subsets):
    gini_index = 0
    total_instances = sum(len(Y) for Y in Y_subsets)
    classes = sorted(set([j for i in Y_subsets for j in i]))

    for Y in Y_subsets:
        m = len(Y)
        if m == 0:
            continue
        count = [Y.count(c) for c in classes]
        gini = 1.0 - sum((n / m) ** 2 for n in count)
        gini_index += (m / total_instances)*gini
    
    return gini_index

def split_data_set(data_X, data_Y, feature_index, threshold):
    left_X = []
    right_X = []
    left_Y = []
    right_Y = []
    for i in range(len(data_X)):
        if data_X[i][feature_index] < threshold:
            left_X.append(data_X[i])
            left_Y.append(data_Y[i])
        else:
            right_X.append(data_X[i])
            right_Y.append(data_Y[i])
    
    return left_X, left_Y, right_X, right_Y


def get_best_split(X, Y):
    best_feature=0
    best_threshold=0
    Best_Gini_index=99999
    for i in range(len(X[0])):
        threshold=sorted(set(X[:,i]))
        for j in threshold:
            left_X,left_Y,right_X,right_Y=split_data_set(X,Y,i,j)
            if len(left_X) == 0 or len(right_X) == 0:
                continue
            gini_index=calculate_gini_index([left_Y,right_Y])
            if gini_index < Best_Gini_index:
                Best_Gini_index, best_feature, best_threshold = gini_index, i, j
                
    return best_feature, best_threshold


def construct_tree(X, Y, max_depth, min_size, depth):
    classes=list(set(Y))
    predicted_class=classes[np.argmax([np.sum(Y==i) for i in classes])]
    node=Node(predicted_class,depth)
    #check is pure
    if len(set(Y))==1:
        return node 
        
    # check max depth reached
    if depth >=max_depth :
        return node
        
    #check min subset at node
    if len(Y)<=min_size:
        return node
        
    feature_index,threshold=get_best_split(X,Y)
    
    if feature_index is None or threshold is None:
        return node
        
    node.feature_index=feature_index
    node.threshold=threshold
    
    left_X,left_Y,right_X,right_Y=split_data_set(X,Y,feature_index,threshold)
    
    node.left=construct_tree(np.array(left_X),np.array(left_Y),max_depth,min_size,depth+1)
    node.right=construct_tree(np.array(right_X),np.array(right_Y),max_depth,min_size,depth+1)
    
    return node

def train_model(train_X,train_Y) :
    root=construct_tree(train_X,train_Y,8,1,0)
    filename = 'MODEL_FILE.sav'
    pickle.dump(root, open(filename, 'wb'))
    

if __name__=="__main__":
    X,Y=Import_data()
    X=data_processing(X)
    #Uncomment it when its requirement over)
    print(mean_null_list)
    print(All_mini_maxi_mean_normalize_value)
    train_model(X,Y)