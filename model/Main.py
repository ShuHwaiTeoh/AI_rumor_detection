import model.function_from_original_author as original
import model.TD_RvNN as TD_RvNN
import torch
import time
import datetime
import numpy as np
import sys
import torch.optim as optim

Nepoch = 300
lr = 0.001 #learning rate

if __name__ == "__main__":
    #load data
    tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test = original.loadData()
    #initialize model
    model = TD_RvNN.RvNN()
    #training and testing
    losses = []
    for epoch in range(Nepoch):
        optimizer = optim.Adam(model.params, lr)
        # optimizer = optim.SGD(model.params, lr, momentum=0.9)
        for i in range(len(y_train)):
            model.zeroGrad()
            pred_y = model.compute_tree(word_train[i], index_train[i], parent_num_train[i], tree_train[i])
            loss = torch.sum((torch.sub(torch.FloatTensor(y_train[i]),pred_y))**2)
            loss.backward()
            optimizer.step()
            losses.append(np.round(loss.detach(),2))
        print("epoch: {}, loss: {}".format(epoch, np.mean(losses)))
        sys.stdout.flush()
        ## calculate loss and evaluate
        if epoch % 5 == 0:
           time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           print("{}: epoch:{}, loss:{}".format(time, epoch, np.mean(losses)))
           sys.stdout.flush()
           prediction = []
           for j in range(len(y_test)):
               pred_y = model.compute_tree(word_test[j], index_test[j], parent_num_test[j], tree_test[j])
               prediction.append(pred_y)
           result = original.evaluation_4class(prediction, y_test)
           print('results: {}'.format(result))
           sys.stdout.flush()

