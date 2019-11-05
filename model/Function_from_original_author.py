import torch
import model.TD_RvNN as TD_RvNN
import numpy as np
import sys

dataset = "Twitter16" # choose dataset, you can choose either "Twitter15" or "Twitter16"
fold = "4" # fold index, choose from 0-4

treePath = '../resource/data.TD_RvNN.vol_5000.txt'
trainPath = "../nfold/RNNtrainSet_"+dataset+str(fold)+"_tree.txt"
testPath = "../nfold/RNNtestSet_"+dataset+str(fold)+"_tree.txt"
labelPath = "../resource/"+dataset+"_label_All.txt"

################################### tools #####################################
class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = [] #children of current node
        self.idx = idx #eid
        self.word = [] #wordFrequent
        self.index = [] #wordIndex
        self.parent = None #parent of current node
        
# generate tree structure
def gen_nn_inputs(root_node, ini_word):
    """Given a root node, returns the appropriate inputs to NN.
    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.
    """
    tree = [[0, root_node.idx]]
    X_word, X_index = [root_node.word], [root_node.index]
    internal_tree, internal_word, internal_index  = _get_tree_path(root_node)
    tree.extend(internal_tree)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    X_word.append(ini_word)
    """
    tree: list of eid list in tree order
    X_word: list of word frequent list in tree order
    X_index: list of word index list in tree order
    """
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))

def _get_tree_path(root_node):
    """Get computation order of leaves -> root."""
    if root_node.children is None:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:]) #[[root node], [nodes in 1st layer...], [nodes in 2nd layer...],...]
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer] #1st iter: [nodes in 1st layer], 2nd iter: [nodes in 2nd layer]
        layer = next_layer #1st iter: [root], 2nd iter: [nodes in 1st layer], ...
    tree = []
    word = []
    index = []
    for layer in layers:
        for node in layer:
            if not node.children:
               continue 
            for child in node.children:
                tree.append([node.idx, child.idx]) # [[1st child eid, 1st child index in tree], [2nd child eid, 2nd child index in tree]...]
                word.append(child.word if child.word is not None else -1) # [[wordFreq of 1st child], [wordFreq of 2nd child], ...]
                index.append(child.index if child.index is not None else -1)# [[wordIndex of 1st child], [wordIndex of 2nd child], ...]
    return tree, word, index

def str2matrix(Str, MaxL): # str is wordIndex : wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [ 0 ]*( MaxL-l ) #padding 0 to max length
    wordFreq += ladd 
    wordIndex += ladd
    return wordFreq, wordIndex 

def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
       y_train = [1,0,0,0]
       l1 += 1 #counter of lable_nonR
    elif label in labelset_f:
       y_train = [0,1,0,0] 
       l2 += 1 #counter of lable_f
    elif label in labelset_t:
       y_train = [0,0,1,0] 
       l3 += 1 
    elif label in labelset_u:
       y_train = [0,0,0,1] 
       l4 += 1
    return y_train, l1,l2,l3,l4

def constructTree(tree):
    ## tree: {index1:{'parent':, 'maxL':, 'vec':}
    ## 1. ini tree node: create a node with each eid in the tree dictionary
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree: fill each node of tree with information from tree dictionary
    for j in tree:
        eidC = j
        indexP = tree[j]['parent']
        nodeC = index2node[eidC]
        wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
           nodeP = index2node[int(indexP)]
           nodeC.parent = nodeP
           nodeP.children.append(nodeC)
        ## root node ##
        else:
           root = nodeC
    ## 3. convert tree to DNN input    
    parent_num = tree[j]['parent_num']
    iniVec, _ = str2matrix( "0:0", tree[j]['maxL'] )
    x_word, x_index, tree = TD_RvNN.gen_nn_inputs(root, iniVec)
    """
    tree: list of eid list in tree order
    X_word: list of word frequent list in tree order
    X_index: list of word index list in tree order
    """
    return x_word, x_index, tree, parent_num

def loadData():
    print("loading tree label")
    labelDic = {}
    with open(labelPath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip().split('\t')
            label, eid = line[0], line[2]
            labelDic[eid] = label.lower()
    print("labelDict length: {}".format(len(labelDic)))
    
    print("reading tree")
    treeDic = {}
    with open(treePath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip().split('\t')
            eid, indexP, indexC, parent_num, maxL, Vec= line[0], line[1], \
                                                int(line[2]), int(line[3]),\
                                                int(line[4]), line[5]
            if eid not in treeDic:
               treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':maxL, 'vec':Vec}
    print('tree length: {}'.format(len(treeDic)))
    
    print("loading train set")
    tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
    l1,l2,l3,l4 = 0,0,0,0
    with open(trainPath, 'r') as f:
        for eid in f.readlines():
            eid = eid.rstrip()
            if eid not in labelDic: continue
            if eid not in treeDic: continue
            if len(treeDic[eid]) <= 0: continue
            ## 1. load label
            label = labelDic[eid]
            y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
            y_train.append(y)
            ## 2. construct tree
            x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
            tree_train.append(tree)
            word_train.append(x_word)
            index_train.append(x_index)
            parent_num_train.append(parent_num)
    
    print("loading test set")
    tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
    l1,l2,l3,l4 = 0,0,0,0
    for eid in open(testPath):
        eid = eid.rstrip()
        if eid not in labelDic: continue
        if eid not in treeDic: continue
        if len(treeDic[eid]) <= 0:
           continue        
        ## 1. load label        
        label = labelDic[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        ## 2. construct tree
        x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
        tree_test.append(tree)
        word_test.append(x_word)  
        index_test.append(x_index) 
        parent_num_test.append(parent_num)
    return tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test

def evaluation_4class(prediction, y): # 4 dim
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0 #news, nonRumor
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0 #fake news
    TP3, FP3, FN3, TN3 = 0, 0, 0, 0 #true news
    TP4, FP4, FN4, TN4 = 0, 0, 0, 0 #nuverified
    e, RMSE, RMSE1, RMSE2, RMSE3, RMSE4 = 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(y)):
        y_i, p_i = list(y[i]), list(prediction[i])
        ##RMSE
        for j in range(len(y_i)):
            RMSE += (y_i[j]-p_i[j])**2
        RMSE1 += (y_i[0]-p_i[0])**2 
        RMSE2 += (y_i[1]-p_i[1])**2 
        RMSE3 += (y_i[2]-p_i[2])**2 
        RMSE4 += (y_i[3]-p_i[3])**2 

        Act = str(y_i.index(max(y_i))+1)  #actual value
        Pre = str(p_i.index(max(p_i))+1)  #pridict value

        ## for class 1
        if Act == '1' and Pre == '1': TP1 += 1
        if Act == '1' and Pre != '1': FN1 += 1
        if Act != '1' and Pre == '1': FP1 += 1
        if Act != '1' and Pre != '1': TN1 += 1
        ## for class 2
        if Act == '2' and Pre == '2': TP2 += 1
        if Act == '2' and Pre != '2': FN2 += 1
        if Act != '2' and Pre == '2': FP2 += 1
        if Act != '2' and Pre != '2': TN2 += 1
        ## for class 3
        if Act == '3' and Pre == '3': TP3 += 1
        if Act == '3' and Pre != '3': FN3 += 1
        if Act != '3' and Pre == '3': FP3 += 1
        if Act != '3' and Pre != '3': TN3 += 1
        ## for class 4
        if Act == '4' and Pre == '4': TP4 += 1
        if Act == '4' and Pre != '4': FN4 += 1
        if Act != '4' and Pre == '4': FP4 += 1
        if Act != '4' and Pre != '4': TN4 += 1
    ## print result
    Acc_all = round( float(TP1+TP2+TP3+TP4)/float(len(y)+e), 4 )
    Acc1 = round( float(TP1+TN1)/float(TP1+TN1+FN1+FP1+e), 4 )
    Prec1 = round( float(TP1)/float(TP1+FP1+e), 4 )
    Recll1 = round( float(TP1)/float(TP1+FN1+e), 4 )
    F1 = round( 2*Prec1*Recll1/(Prec1+Recll1+e), 4 )
    
    Acc2 = round( float(TP2+TN2)/float(TP2+TN2+FN2+FP2+e), 4 )
    Prec2 = round( float(TP2)/float(TP2+FP2+e), 4 )
    Recll2 = round( float(TP2)/float(TP2+FN2+e), 4 )
    F2 = round( 2*Prec2*Recll2/(Prec2+Recll2+e), 4 )
    
    Acc3 = round( float(TP3+TN3)/float(TP3+TN3+FN3+FP3+e), 4 )
    Prec3 = round( float(TP3)/float(TP3+FP3+e), 4 )
    Recll3 = round( float(TP3)/float(TP3+FN3+e), 4 )
    F3 = round( 2*Prec3*Recll3/(Prec3+Recll3+e), 4 )
    
    Acc4 = round( float(TP4+TN4)/float(TP4+TN4+FN4+FP4+e), 4 )
    Prec4 = round( float(TP4)/float(TP4+FP4+e), 4 )
    Recll4 = round( float(TP4)/float(TP4+FN4+e), 4 )
    F4 = round( 2*Prec4*Recll4/(Prec4+Recll4+e), 4 )
    
    microF = round( (F1+F2+F3+F4)/4,5 )
    n_digits = 4
    RMSE_all = torch.round( ( RMSE/len(y) )**0.5*10**n_digits) / (10**n_digits)
    RMSE_all_1 = torch.round( ( RMSE1/len(y) )**0.5*10**n_digits) / (10**n_digits)
    RMSE_all_2 = torch.round( ( RMSE2/len(y) )**0.5*10**n_digits) / (10**n_digits)
    RMSE_all_3 = torch.round( ( RMSE3/len(y) )**0.5*10**n_digits) / (10**n_digits)
    RMSE_all_4 = torch.round( ( RMSE4/len(y) )**0.5*10**n_digits) / (10**n_digits)
    RMSE_all_avg = torch.round( ( RMSE_all_1+RMSE_all_2+RMSE_all_3+RMSE_all_4 )/4*10**n_digits) / (10**n_digits)
    return ['acc:',Acc_all, 'Favg:',microF, #RMSE_all, RMSE_all_avg,
            'C1:',Acc1, Prec1, Recll1, F1,
            'C2:',Acc2, Prec2, Recll2, F2,
            'C3:',Acc3, Prec3, Recll3, F3,
            'C4:',Acc4, Prec4, Recll4, F4]



