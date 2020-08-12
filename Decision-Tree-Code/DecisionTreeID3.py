import numpy as np
import pandas as pd
import math

max_label = -1

def dataclassify(data):
    column_label=data[:,-1]
    unique_classes,count_of_unique_classes=np.unique(column_label,return_counts=True)
    index=count_of_unique_classes.argmax()
    classification=unique_classes[index]
    return classification

def getEntropy(data):
    column_label=data[:,-1]
    unique_classes,count_of_unique_classes=np.unique(column_label,return_counts=True)
    if len(unique_classes)==1:
        return 0
    entropy = 0
    sum_total = count_of_unique_classes.sum()
    for i in list(count_of_unique_classes):
        p = float(i)/sum_total
        entropy -= p*math.log(p,2)
    return entropy

def getInformationGain(data,attribute):
    Split_column_values=data[:,attribute]
    d_positive=data[Split_column_values==1]
    d_negative=data[Split_column_values==0]
    n=float(len(d_positive)+len(d_negative))
    p_positive=float(len(d_positive)/n)
    p_negative=float(len(d_negative)/n)
    InformationGain=getEntropy(data)-p_positive*getEntropy(d_positive)-p_negative*getEntropy(d_negative)
    return InformationGain

def getbest_attribute(data):
    Info_Gain=-1
    best_attribute = -1
    _,no_of_attributes=data.shape
    for i in range(no_of_attributes-1):
        #if(visited[i]==0):
            InfoGain_Present_Attribute=getInformationGain(data,i)
            if(InfoGain_Present_Attribute>Info_Gain):
                Info_Gain=InfoGain_Present_Attribute
                best_attribute=i
    return best_attribute

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def ID3(data,visited):
    #root=Node()
    #temp=root
    label_column=data[:,-1]
    unique_classes=np.unique(label_column)
    no_of_instances,_=data.shape
    if no_of_instances==0:
        root=Node(max_label)
        #temp.val=max_label
        #temp.left=None
        #temp.right=None
        return root
    if len(unique_classes)==1:
        root=Node(unique_classes[0])
        #temp.val=unique_classes[0]
        #temp.left=None
        #temp.right=None
        return root
    if (visited.count(0) == 1):
        root=Node(dataclassify(data)) 
        #temp.val=dataclassify(data)
        #temp.left=None
        #temp.right=None
        return root  
    else:
        bestattribute=getbest_attribute(data)
        root=Node(bestattribute)
        #temp.val=bestattribute
        visited[root.val]=1
        split_column_values=data[:,bestattribute]
        data_positive=data[split_column_values==1]
        data_negative=data[split_column_values==0]
        root.left=ID3(data_positive,visited)
        root.right=ID3(data_negative,visited)
        visited[root.val]=0
        return root

def getVariance(data):
    column_label=data[:,-1]
    unique_classes,count_of_unique_classes=np.unique(column_label,return_counts=True)
    if len(unique_classes)==1:
        return 0
    else:
        variance = 1
        sum_total = count_of_unique_classes.sum()
        for i in list(count_of_unique_classes):
            variance *= float(i)/float(sum_total)
        
    return variance

def getInformationGain_V(data,attribute):
    Split_column_values=data[:,attribute]
    d_positive=data[Split_column_values==1]
    d_negative=data[Split_column_values==0]
    n=float(len(d_positive)+len(d_negative))
    p_positive=float(len(d_positive)/n)
    p_negative=float(len(d_negative)/n)
    InformationGain_V=getVariance(data)-p_positive*getVariance(d_positive)-p_negative*getVariance(d_negative)
    return InformationGain_V

def getbest_attribute_V(data):
    Info_Gain_V=-1
    best_attribute_V = -1
    _,no_of_attributes=data.shape
    for i in range(no_of_attributes-1):
        #if(visited[i]==0):
            InfoGain_Present_Attribute_V=getInformationGain_V(data,i)
            if(InfoGain_Present_Attribute_V>Info_Gain_V):
                Info_Gain_V=InfoGain_Present_Attribute_V
                best_attribute_V=i
    return best_attribute_V

def ID3_V(data,visited):
    #root=Node()
    #temp=root
    label_column=data[:,-1]
    unique_classes=np.unique(label_column)
    no_of_instances,_=data.shape
    if no_of_instances==0:
        root=Node(max_label)
        #temp.val=max_label
        #temp.left=None
        #temp.right=None
        return root
    if len(unique_classes)==1:
        root=Node(unique_classes[0])
        #temp.val=unique_classes[0]
        #temp.left=None
        #temp.right=None
        return root
    if (visited.count(0) == 1):
        root=Node(dataclassify(data)) 
        #temp.val=dataclassify(data)
        #temp.left=None
        #temp.right=None
        return root  
    else:
        bestattribute=getbest_attribute_V(data)
        root=Node(bestattribute)
        #temp.val=bestattribute
        visited[root.val]=1
        split_column_values=data[:,bestattribute]
        data_positive=data[split_column_values==1]
        data_negative=data[split_column_values==0]
        root.left=ID3_V(data_positive,visited)
        root.right=ID3_V(data_negative,visited)
        visited[root.val]=0
        return root

def TreeTraverse(root,data,row_no):
    
    if(root.left==None and root.right==None):
        return root.val
    if(data[row_no,root.val]==1):
        if(root.left!=None):
            return TreeTraverse(root.left,data,row_no)
        else:
            return max_label
    if(data[row_no,root.val]==0):
        if(root.right!=None):
            return TreeTraverse(root.right,data,row_no)
        else:
            return max_label

def GetAccuracy(root,data):
    no_of_instances,_=data.shape
    #print(no_of_instances)
    count=0
    for i in range(no_of_instances): 
        leaf_tree=TreeTraverse(root,data,i)
        if(leaf_tree==data[i,-1]):
            count=count+1
    accuracy=(float(count)/float(no_of_instances))*100
    return accuracy

def BuildTree_Depth_Prune(data, root, depth):
   
    if(root.left == None and root.right == None):
        root_prune=Node(root.val)
        return root_prune
    
    elif(depth == 0):
        root_prune = Node(dataclassify(data))
        return root_prune
    
    else:
        split_column_values = data[:,root.val]
        data_positive = data[split_column_values == 1]
        data_negative = data[split_column_values == 0]
        root_prune = Node(root.val)
        root_prune.left = BuildTree_Depth_Prune(data_positive, root.left, depth-1)
        root_prune.right = BuildTree_Depth_Prune(data_negative, root.right, depth-1)
        return root_prune

def copy_subtree(node):
    copy_node = Node(node.val)
   
    if(node.left == None and node.right == None):
        return copy_node
    
    copy_node.left = copy_subtree(node.left)
    copy_node.right = copy_subtree(node.right)
    
    return copy_node

def DepthBasedPruning(root,data,v_data):
        
    dmax=[5,10,15,20,50,100]
    accuracy_v=0
    for i in dmax:
        pruneroot=BuildTree_Depth_Prune(data,root,i)
        prunedtree_accuracy=GetAccuracy(pruneroot,v_data)
        if (prunedtree_accuracy>accuracy_v):
            accuracy_v=prunedtree_accuracy
            prunedtree=copy_subtree(pruneroot)
    return prunedtree     


def find_depth(root_copy, pr_depth):
    
    if(root_copy.left == None and root_copy.right == None):
        return pr_depth
    
    if(root_copy.left!=None):
        left_depth = find_depth(root_copy.left, pr_depth+1)
        
    if(root_copy.right!=None):
        right_depth = find_depth(root_copy.right, pr_depth+1)
        
    return max(left_depth, right_depth)


def get_Nodes_List_level(root_copy, level, Nodes_List):
    
    if(root_copy.left == None and root_copy.right == None):
        return Nodes_List
    
    if(level == 0):
        Nodes_List.append(root_copy.val)
        return Nodes_List
    
    Nodes_List = get_Nodes_List_level(root_copy.left, level-1, Nodes_List)
    Nodes_List = get_Nodes_List_level(root_copy.right, level-1, Nodes_List)
    
    return Nodes_List
    

def getNodes_List(root_copy, Nodes_List):
    
    depth = find_depth(root_copy, 0)
   
    for i in range(depth):
        Nodes_List = get_Nodes_List_level(root_copy,i, Nodes_List)
    
    return Nodes_List


def ReplaceNode(root_copy, node_val, data, visited_reduced_prune, i):
    if(root_copy.left == None and root_copy.right == None):
        return root_copy
    
    if(root_copy.val == node_val and visited_reduced_prune[i] == 0):
        no_of_instances, no_of_attributes = data.shape

        if(no_of_instances!=0):
            new_node = dataclassify(data)
            root_copy.val = new_node
            root_copy.left = None
            root_copy.right = None
            return root_copy
        else:
            new_node = max_label
            root_copy.val = new_node
            root_copy.left = None
            root_copy.right = None
            return root_copy
    else:
        split_column_values = data[:,root_copy.val]
        data_positive = data[split_column_values == 1]
        data_negative = data[split_column_values == 0]
        if(root_copy.left != None):
            root_copy.left = ReplaceNode(root_copy.left, node_val, data_positive, visited_reduced_prune,i)
        
        if(root_copy.right != None):
            root_copy.right = ReplaceNode(root_copy.right, node_val, data_negative, visited_reduced_prune,i)
        
        return root_copy


def BuildTree_Reduced_Error(data, data_valid, root):
    #lobal Nodes_List
    root_copy = copy_subtree(root)
    final_copy_tree = copy_subtree(root)
    Nodes_List = []
    Nodes_List = getNodes_List(root_copy, Nodes_List)
    visited_reduced_prune = [0]*len(Nodes_List)
    #print(Nodes_List)
    #print(visited_reduced_prune)
    present_best_accuracy = GetAccuracy(final_copy_tree,data_valid.values)
    #print(present_best_accuracy)
    #print(getAccuracy(data_valid, root))
    for i in reversed(range(len(Nodes_List))):
        root_copy = copy_subtree(final_copy_tree)
        root_copy = ReplaceNode(root_copy, Nodes_List[i], data, visited_reduced_prune, i)
        visited_reduced_prune[i] = 1
        if(GetAccuracy(root_copy,data_valid.values) > present_best_accuracy):
            final_copy_tree = copy_subtree(root_copy)
        #print(i)
            
    return final_copy_tree
    
