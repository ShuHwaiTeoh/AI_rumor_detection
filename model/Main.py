import model.function_from_original_author as original
import model.TD_RvNN as TD_RvNN
import torch
import time
import datetime
import numpy as np
import sys
import torch.optim as optim

Nepoch = 600
lr = 0.001 #learning rate

if __name__ == "__main__":
    ## 1. load tree & word & index & label
    tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test = original.loadData()
    ## 2. ini RNN model
    t0 = time.time()
    model = TD_RvNN.RvNN()
    t1 = time.time()
    print('Recursive model established,', (t1-t0)/60)
    ## 3. looping SGD
    losses_5, losses = [], []
    num_examples_seen = 0
    for epoch in range(Nepoch):
        #optimizer = optim.Adam(model.params, lr)
        optimizer = optim.SGD(model.params, lr, momentum=0.9)
        for i in range(len(y_train)):
            model.zeroGrad()
            pred_y = model.compute_tree(word_train[i], index_train[i], parent_num_train[i], tree_train[i])
            loss = torch.sum((torch.sub(torch.FloatTensor(y_train[i]),pred_y))**2)
            loss.backward()
            optimizer.step()
            losses.append(np.round(loss.detach(),2))
            num_examples_seen += 1
        print("epoch=%d: loss=%f" % ( epoch, np.mean(losses) ))
        sys.stdout.flush()
        ## cal loss & evaluate
        if epoch % 5 == 0:
           losses_5.append((num_examples_seen, np.mean(losses)))
           time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, np.mean(losses)))
           sys.stdout.flush()
           prediction = []
           for j in range(len(y_test)):
               pred_y = model.compute_tree(word_test[j], index_test[j], parent_num_test[j], tree_test[j])
               prediction.append(pred_y)
           res = original.evaluation_4class(prediction, y_test)
           print('results:', res)
           sys.stdout.flush()

