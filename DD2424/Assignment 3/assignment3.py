import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import sys

#for dict destructuring
dictget = lambda d, *k: [d[i] for i in k]

np.random.seed(42) #ensure same seed per run

def load_file(filename):
    with open("Datasets/"+filename, 'rb') as x:
        dict = pickle.load(x, encoding = 'bytes')
        return dict
    
def load_batch(filename):
    dict = load_file(filename)
    #https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
    #explanation: for each k, we replace k with eye(n)[k], from value to row with 1 at kth index
    return dict[b"data"].T.astype("double"), np.eye(10)[dict[b"labels"]].T, dict[b"labels"]

def normalise(X):
    #across all samples, then reshape to column vector
    X_mean = np.mean(X,axis=1).reshape(-1,1)
    X_std = np.std(X,axis=1).reshape(-1,1)
    X -= X_mean
    X /= X_std
    return X

def initialise(node_count,He=False,std=None,batch_norm=False):
    K = 10
    d = 3072
    W = []
    B = []
    
    n_layers = len(node_count)
    
    #initialise 1st layer
    if He:
        W.append(np.random.normal(0, np.sqrt(2/d), size = (node_count[0],d)))
    elif std:
        W.append(np.random.normal(0, std, size = (node_count[0],d)))
    else:
        raise Exception("Please specify either He initialisation or a standard deviation")
    
    B.append(np.zeros((node_count[0],1)))
    
    for i in range(1,n_layers):
        if He:
            W.append(np.random.normal(0, np.sqrt(2/node_count[i-1]), size = (node_count[i],node_count[i-1])))
        else:
            W.append(np.random.normal(0, std, size = (node_count[i],node_count[i-1])))
        B.append(np.zeros((node_count[i],1)))

    #later: gamma and beta for batch norm
    if batch_norm:
        gamma = []
        beta = []
        for i in range(n_layers-1):
            gamma.append(np.ones((node_count[i], 1)))
            beta.append(np.zeros((node_count[i], 1)))
        return W,B,gamma,beta
    
    return W,B
    

def SoftMax(s):
    return np.exp(s)/np.sum(np.exp(s),axis=0)

def relu(s):
    return np.maximum(0,s)

def wh_plus_b(h,w,b):
    return np.matmul(w,h)+b

def batch_norm_func(scores,mean,var):
    scores_norm = np.matmul(np.diag(1/(var+np.finfo(np.double).eps)**0.5), scores-mean.reshape(-1,1))
    #add eps for stability
    return scores_norm

def EvaluateClassifier(X,W,B,batch_norm=False,gamma=None,beta=None,mean=None,variance=None):
    layers = len(W)
    
    scores = []
    #contains calculations of each layer before relu
    activations = [copy.deepcopy(X)]
    #contains activations of prev layers, to feed into next layer
    score_norms = None
    score_means = None
    score_vars = None
    
    if batch_norm:
        score_norms = []
        score_means = mean if mean else []
        score_vars = variance if variance else []
        for i in range(layers-1):
            scores.append(wh_plus_b(activations[-1],W[i],B[i]))
            if not mean and not variance:
                score_means.append(np.mean(scores[-1], axis = 1, dtype = np.double))
                score_vars.append(np.var(scores[-1], axis = 1, dtype = np.double))
            score_norms.append(batch_norm_func(scores[-1],score_means[i],score_vars[i]))
            activations.append(relu(np.multiply(gamma[i],score_norms[-1]) + beta[i]))
        scores.append(wh_plus_b(activations[-1],W[-1],B[-1]))
        activations.append(SoftMax(scores[-1]))
    else:
        for i in range(layers):
            scores.append(wh_plus_b(activations[-1],W[i],B[i]))
            if i != (layers-1): #if last layer, softmax instead
                activations.append(relu(scores[-1]))
        activations.append(SoftMax(scores[-1]))

    return {'scores':scores,
            'activations':activations,
            'score_norms':score_norms,
            'score_means':score_means,
            'score_vars':score_vars}

def ComputeLoss(X,Y,W,B,batch_norm=False,gamma=None,beta=None,mean=None,variance=None):
    p = EvaluateClassifier(X,W,B,batch_norm,gamma,beta,mean,variance)['activations'][-1]
    return (1/X.shape[1])*np.sum(-np.log((np.sum(Y*p, axis=0))))

def ComputeCost(X,Y,W,B,lamda,batch_norm=False,gamma=None,beta=None,mean=None,variance=None):
    return ComputeLoss(X,Y,W,B,batch_norm,gamma,beta,mean,variance)+lamda*np.sum([np.sum(np.square(w)) for w in W])
    #find ComputeLoss
    #and then add L2 term

def ComputeAccuracy(X,y,W,B,batch_norm=False,gamma=None,beta=None,mean=None,variance=None):
    p = EvaluateClassifier(X,W,B,batch_norm,gamma,beta,mean,variance)['activations'][-1]
    predictions = np.argmax(p, axis=0)
    return (y==predictions).sum()/len(y)
    #EvaluateClassifier
    #argmax along axis 0
    #divide sum of matching prediction over length to get accuracy


def batch_norm_back_pass(g,scores,mean,variance):
    n = scores.shape[1]
    eps = np.finfo(np.double).eps
    sig_1 = np.power(variance+eps,-0.5).T.reshape(-1,1)
    sig_2 = np.power(variance+eps,-1.5).T.reshape(-1,1)
    g_1 = g * np.matmul(sig_1,np.ones((1,n), dtype="double"))
    g_2 = g * np.matmul(sig_2,np.ones((1,n), dtype="double"))
    D = scores - np.matmul(mean.reshape(-1,1),np.ones((1,n), dtype="double"))
    c = np.matmul(np.multiply(g_2,D), np.ones((n,1), dtype="double"))
    return g_1 - 1/n * np.matmul(np.matmul(g_1,np.ones((n,1), dtype="double")),np.ones((1,n), dtype="double")) - 1/n*np.multiply(D,np.matmul(c,np.ones((1,n), dtype="double")))



def ComputeGrads(Y,activations,W,B,lamda,batch_norm=False,gamma=None,scores=None,score_norms=None,score_means=None,score_vars=None):

    #number of entries in batch, i.e cols in X
    n = activations[0].shape[1]
    P = activations[-1]

    #initial gradient
    g = -(Y-P)
    
    grad_W = []
    grad_B = []
    grad_beta = None
    grad_gamma = None
    if not batch_norm:
        for i in reversed(range(len(W))):
            grad_W.append(1/n*np.matmul(g,activations[i].T)+2*lamda*W[i])
            grad_B.append(1/n*np.matmul(g,np.ones((n,1), dtype="double")))
            g = np.matmul(W[i].T,g)
            h = copy.deepcopy(activations[i])
            h[h>np.double()] = np.double(1)
            g = np.multiply(g,h)

        grad_W.reverse()
        grad_B.reverse()
        
    else: 
        grad_gamma = []
        grad_beta = []
        grad_W.append(1/n*np.matmul(g, activations[-2].T)+2*lamda*W[-1])
        grad_B.append(1/n*np.matmul(g,np.ones((n,1), dtype="double")))
        g = np.matmul(W[-1].T,g)
        h = copy.deepcopy(activations[-2])
        h[h>np.double()] = np.double(1)
        g = np.multiply(g,h)
        for i in reversed(range(len(W)-1)):
            grad_gamma.append(1/n*np.matmul(np.multiply(g,score_norms[i]),np.ones((n,1), dtype="double")))
            grad_beta.append(1/n*np.matmul(g,np.ones((n,1), dtype="double")))
            g = np.multiply(g,np.matmul(gamma[i],np.ones((1,n), dtype="double")))
            g = batch_norm_back_pass(g,scores[i],score_means[i],score_vars[i])
            grad_W.append(1/n*np.matmul(g, activations[i].T)+2*lamda*W[i])
            grad_B.append(1/n*np.matmul(g,np.ones((n,1), dtype="double")))
            if i != 0:
                g = np.matmul(W[i].T,g)
                h = copy.deepcopy(activations[i])
                h[h>np.double()] = np.double(1)
                g = np.multiply(g,h)
        
        grad_W.reverse()
        grad_B.reverse()
        grad_beta.reverse()
        grad_gamma.reverse()
    
    return {'grad_W': grad_W,
            'grad_B': grad_B,
            'grad_beta': grad_beta,
            'grad_gamma': grad_gamma}



def ComputeGradsNumSlow(Y, activations, W, B, lamda, h, batch_norm=False, gamma=None, beta=None):
    """ Converted from matlab code """

    grad_W = [np.zeros(x.shape) for x in W]
    grad_B = [np.zeros(x.shape) for x in B]
    
    if gamma and beta:
        grad_gamma = [np.zeros(x.shape) for x in gamma]
        grad_beta = [np.zeros(x.shape) for x in beta]
    
    for j in range(len(B)):
        for i in range(len(B[j])):
            b_try = [np.array(x) for x in B]
            b_try[j][i] -= h
            c1 = ComputeCost(activations[0], Y, W, b_try, lamda, batch_norm, gamma, beta)

            b_try = [np.array(x) for x in B]
            b_try[j][i] += h
            c2 = ComputeCost(activations[0], Y, W, b_try, lamda, batch_norm, gamma, beta)
            grad_B[j][i] = (c2-c1) / (2*h)
            

    for k in range(len(W)):
        for j in range(W[k].shape[0]):
            for i in range(W[k].shape[1]):
                W_try = [np.array(x) for x in W]
                W_try[k][j][i] -= h
                c1 = ComputeCost(activations[0], Y, W_try, B, lamda, batch_norm, gamma, beta)
                grad_W[k][j][i] = (c2-c1) / (2*h)
            
                W_try = [np.array(x) for x in W]
                W_try[k][j][i] += h
                c2 = ComputeCost(activations[0], Y, W_try, B, lamda, batch_norm, gamma, beta)
                grad_W[k][j][i] = (c2-c1) / (2*h)

    for j in range(len(beta)):
        for i in range(len(beta[j])):
            beta_try = [np.array(x) for x in beta]
            beta_try[j][i] -= h
            c1 = ComputeCost(activations[0], Y, W, B, lamda, batch_norm, gamma, beta_try)

            beta_try = [np.array(x) for x in beta]
            beta_try[j][i] += h
            c2 = ComputeCost(activations[0], Y, W, B, lamda, batch_norm, gamma, beta_try)
            grad_beta[j][i] = (c2-c1) / (2*h)

    for j in range(len(gamma)):
        for i in range(len(gamma[j])):
            gamma_try = [np.array(x) for x in gamma]
            gamma_try[j][i] -= h
            c1 = ComputeCost(activations[0], Y, W, B, lamda, batch_norm, gamma_try, beta)

            gamma_try = [np.array(x) for x in gamma]
            gamma_try[j][i] += h
            c2 = ComputeCost(activations[0], Y, W, B, lamda, batch_norm, gamma_try, beta)
            grad_gamma[j][i] = (c2-c1) / (2*h)


    return {'grad_W': grad_W,
            'grad_B': grad_B,
            'grad_beta': grad_beta,
            'grad_gamma': grad_gamma}



def compareGrads(X, Y, W, B, lamda, h=1e-5, batch_norm=False,gamma=None,beta=None):
    n = X.shape[1]
    print('Number of entries: ', n)
    print('Lambda: ', lamda, '\n')
    scores, activations, score_norms, score_means, score_vars = dictget(
        EvaluateClassifier(X,W,B,batch_norm,gamma,beta),
        'scores','activations','score_norms','score_means','score_vars')

    def findRelativeError(gradA, gradB):
        grad_shape = gradA.shape
        relative_error = np.divide(np.abs(gradA-gradB),np.maximum(np.finfo(np.float32).eps, np.abs(gradA)+np.abs(gradB)))
        print(f'Mean relative error: {np.mean(relative_error)}')
        return(np.allclose(relative_error, np.zeros(grad_shape,dtype="double"),rtol=0.0,atol=1e-3))
    
    grad_W, grad_B, grad_beta, grad_gamma = dictget(
        ComputeGrads(Y, activations, W, B, lamda, batch_norm, gamma, scores, score_norms, score_means, score_vars),
        'grad_W','grad_B','grad_beta','grad_gamma')
    
    #check with num

    print('Comparing with slow numerical method \n')
    grad_W_num_slow ,grad_B_num_slow, grad_beta_num_slow, grad_gamma_num_slow = dictget(
        ComputeGradsNumSlow(Y, activations, W, B, lamda, h, batch_norm, gamma, beta),
        'grad_W','grad_B','grad_beta','grad_gamma')
    
    print([x.shape for x in grad_W])
    print([x.shape for x in grad_W_num_slow])
    for i in range(len(grad_W)):

        print(f'Error for grad wrt W{i}: ', not findRelativeError(grad_W[i], grad_W_num_slow[i]))
        print(f'Error for grad wrt b{i}: ', not findRelativeError(grad_B[i], grad_B_num_slow[i]))
        if batch_norm and i != (len(grad_W)-1):
            print(f'Error for grad wrt beta{i}: ', not findRelativeError(grad_beta[i], grad_beta_num_slow[i]))
            print(f'Error for grad wrt gamma{i}: ', not findRelativeError(grad_gamma[i], grad_gamma_num_slow[i]), '\n')

    print('------------------------------------------------ \n')


def MiniBatchGD(X, Y, y, validX, validY, validy, n_batch, eta, n_epochs, W, B, lamda, batch_norm=False, gamma=None, beta=None, alpha=None):
    Wstar = copy.deepcopy(W)
    Bstar = copy.deepcopy(B)
    gammastar = copy.deepcopy(gamma)
    betastar = copy.deepcopy(beta)
    means = None
    variances = None

    n = X.shape[1]

    t = 0

    training_log = pd.DataFrame(columns = ['train_loss','train_cost','train_acc','val_loss','val_cost','val_acc','eta'])
    for i in range(n_epochs):
        for j in range(n//n_batch):
            eta_t = eta(t)
            start = j*n_batch
            end = (j+1)*n_batch
            X_batch = X[:,start:end]
            Y_batch = Y[:,start:end]
            scores, activations, score_norms, score_means, score_vars = dictget(
                EvaluateClassifier(X_batch,Wstar,Bstar,batch_norm,gammastar,betastar),
                'scores','activations','score_norms','score_means','score_vars')
            if batch_norm:
                if means == None and variances == None:
                    means, variances = score_means, score_vars
                else:
                    for z in range(len(means)):
                        means[z] = (alpha*means[z]+(1-alpha)*score_means[z])
                        variances[z] = (alpha*variances[z]+(1-alpha)*score_vars[z])
            grad_W, grad_B, grad_beta, grad_gamma = dictget(
                ComputeGrads(Y_batch, activations, Wstar, Bstar, lamda, batch_norm, gammastar, scores, score_norms, score_means, score_vars),
                'grad_W','grad_B','grad_beta','grad_gamma')
            for k in range(len(Wstar)):
                Wstar[k] -= eta_t*grad_W[k]
                Bstar[k] -= eta_t*grad_B[k]
                if batch_norm and k != (len(Wstar)-1):
                    betastar[k] -= eta_t*grad_beta[k]
                    gammastar[k] -= eta_t*grad_gamma[k]
            if (t%(n//n_batch*n_epochs//100)) == 0:
                train_loss = ComputeLoss(X,Y,Wstar,Bstar,batch_norm,gammastar,betastar)
                valid_loss = ComputeLoss(validX,validY,Wstar,Bstar,batch_norm,gammastar,betastar)
                training_log.loc[t,'eta'] = eta_t
                training_log.loc[t,'train_loss'] = train_loss
                training_log.loc[t,'train_cost'] = ComputeCost(X,Y,Wstar,Bstar,lamda,batch_norm,gammastar,betastar)
                training_log.loc[t,'train_acc'] = ComputeAccuracy(X,y,Wstar,Bstar,batch_norm,gammastar,betastar,means,variances)
                training_log.loc[t,'val_loss'] = valid_loss
                training_log.loc[t,'val_cost'] = ComputeCost(validX,validY,Wstar,Bstar,lamda,batch_norm,gammastar,betastar)
                training_log.loc[t,'val_acc'] = ComputeAccuracy(validX,validy,Wstar,Bstar,batch_norm,gammastar,betastar,means,variances)
            t += 1
        if i%10 == 0:
            train_loss = ComputeLoss(X,Y,Wstar,Bstar,batch_norm,gammastar,betastar)
            valid_loss = ComputeLoss(validX,validY,Wstar,Bstar,batch_norm,gammastar,betastar)
            print(f'On epoch {i} of {n_epochs}')
            print(f'Training loss: {train_loss}')
            print(f'Validation loss: {valid_loss}')
        
    global image_count
    ax = plt.subplot()
    ax.plot(training_log['train_loss'], label = 'training loss')
    ax.plot(training_log['val_loss'], label = 'validation loss')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.legend()
    ax.set_title(f'cross-entropy loss against step for lamda of {lamda}, batch size of {n_batch}, \n cyclical eta and {n_epochs} epochs')
    plt.savefig(f'{lamda}_lamda_{n_batch}_batches_cyclical_eta_{n_epochs}_epochs_batch_norm_{batch_norm}_loss_{image_count}.png', dpi=300)
    plt.close()
    ax = plt.subplot()
    ax.plot(training_log['train_cost'], label = 'training cost')
    ax.plot(training_log['val_cost'], label = 'validation cost')
    ax.set_xlabel('step')
    ax.set_ylabel('cost')
    ax.legend()
    ax.set_title(f'cross-entropy cost against step for lamda of {lamda}, batch size of {n_batch}, \n cylical eta and {n_epochs} epochs')
    plt.savefig(f'{lamda}_lamda_{n_batch}_batches_cylical_eta_{n_epochs}_epochs_batch_norm_{batch_norm}_cost_{image_count}.png', dpi=300)
    plt.close()
    training_log.to_csv(f'{[x.shape[0] for x in W]}_network_{lamda}_lamda_{n_batch}_batches_cylical_eta_{n_epochs}_epochs_batch_norm_{batch_norm}_cost_{image_count}.csv')
    # ax.plot(training_log['eta'], label = 'eta')
    # ax.set_xlabel('step')
    # ax.set_ylabel('eta')
    # ax.legend()
    # ax.set_title(f'eta against step')
    # plt.savefig(f'eta.png', dpi=300)
    # plt.close()
    image_count += 1
    return {'W':Wstar,
            'B':Bstar,
            'gamma':gammastar,
            'beta':betastar,
            'training_log':training_log,
            'mean':means,
            'var':variances}


def cyclical_eta(stepsize,eta_min,eta_max):
    def eta_func(t):
        if (t//stepsize)%2==0: #in even phase
            eta_t = eta_min + (t%stepsize)/(stepsize)*(eta_max-eta_min)
        else: #in odd phase
            eta_t = eta_max - (t%stepsize)/(stepsize)*(eta_max-eta_min)
        #print(eta_t)
        return eta_t
    return eta_func

def montage(W, filename):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[i*5+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.savefig(filename, dpi=300)


def main():
    global image_count
    image_count = 0
    

    # #uncomment below to check gradients
    # trainX,trainY,trainy = load_batch('data_batch_1')
    # validX,validY,validy = load_batch('data_batch_2')
    # testX,testY,testy = load_batch('test_batch')
    # trainX = normalise(trainX)
    # validX = normalise(validX)
    # testX = normalise(testX)

    # W,B = initialise([50,50,50,10],True)

    # #test with 1st sample, lamda = 0
    # compareGrads(trainX[:20,0].reshape(-1,1),trainY[:,0].reshape(-1,1), [W[i][:,:20] if i==0 else W[i] for i in range(len(W))], B, 0, 1e-4)

    # #test with 100 samples, lamda = 0
    # compareGrads(trainX[:20,:100], trainY[:,:100], [W[i][:,:20] if i==0 else W[i] for i in range(len(W))], B, 0, 1e-4)

    # print('Testing by overfitting to training data with two layers')
    # print('------------------------------------------------ \n')
    # W,B = initialise([50,10],True)
    # n_batch, eta, n_epochs, lamda = 10, lambda x:0.01 , 200, 0
    # W_new, b_new = MiniBatchGD(trainX[:,:100],trainY[:,:100],trainy[:100],validX,validY,validy,n_batch,eta,n_epochs,W,B,lamda)
    # print('Training complete, performing inference on test set')
    # print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,b_new))


    #load all batches
    trainXall,trainYall,trainyall = load_batch('data_batch_1')
    for i in range(2,6):
        trainX_i, trainY_i, trainy_i = load_batch(f'data_batch_{i}')
        trainXall = np.concatenate((trainXall, trainX_i), axis = 1)
        trainYall = np.concatenate((trainYall, trainY_i), axis = 1)
        trainyall.extend(trainy_i)

    #randomly select validation set and remove from train set

    trainXall, validXall, trainYall, validYall, trainyall, validyall = train_test_split(trainXall.T, trainYall.T, trainyall, test_size=0.10, random_state=42)

    trainXall, validXall, trainYall, validYall = trainXall.T, validXall.T, trainYall.T, validYall.T

    testX,testY,testy = load_batch('test_batch')
    trainXall = normalise(trainXall)
    validXall = normalise(validXall)
    testX = normalise(testX)

    #test = EvaluateClassifier(validXall,W,B,True,gamma,beta)

    #for i,j in test.items():
        #print(i)
        #print([k.shape for k in j])

    #ComputeGrads(validYall,test['activations'],W,B,0,True,gamma,test['scores'],test['scores'],test['score_means'],test['score_vars'])

    #uncomment below to check gradients

    W,B,gamma,beta = initialise([50,50,10],True,None,True)

    #gradient test with 100 samples, lamda = 0
    compareGrads(trainXall[:20,:100], trainYall[:,:100], [W[i][:,:20] if i==0 else W[i] for i in range(len(W))], B, 0, 1e-4, True, gamma, beta)
    
    # print('Training with two layers')
    # print('------------------------------------------------ \n')
    # W,B = initialise([50,10],True)
    # lamda = 0.01
    # n_batch = 100
    # cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    # #each epoch has 450 steps, to get 2 cycles: 4*5*450/450 = 20 epochs
    # n_epochs = 20
    # W_new, b_new, _ = MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda)
    # print('Training complete, performing inference on test set')
    # print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,b_new))


    print('Training with three layers without batch normalisation')
    print('------------------------------------------------ \n')
    W,B = initialise([50,50,10],True)
    lamda = 0.005
    n_batch = 100
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    #each epoch has 450 steps, to get 2 cycles: 4*5*450/450 = 20 epochs
    n_epochs = 20
    W_new, B_new, three_layers_no_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda),
        'W','B','training_log')
    print('Training complete, performing inference on test set')
    print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,B_new))


    print('Training with nine layers without batch normalisation')
    print('------------------------------------------------ \n')
    W,B = initialise([50,30,20,20,10,10,10,10,10],True)
    lamda = 0.005
    n_batch = 100
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    n_epochs = 20
    W_new, B_new, nine_layers_no_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda),
        'W','B','training_log')
    print('Training complete, performing inference on test set')
    print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,B_new))

    print('Training with three layers with batch normalisation')
    print('------------------------------------------------ \n')
    W,B, gamma, beta= initialise([50,50,10],True,None,True)
    lamda = 0.005
    n_batch = 100
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    #each epoch has 450 steps, to get 2 cycles: 4*5*450/450 = 20 epochs
    n_epochs = 20
    W_new, B_new, gamma_new, beta_new, last_mean, last_var, three_layers_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda,True,gamma,beta,0.9),
        'W','B','gamma','beta','mean','var','training_log')
    print('Training complete, performing inference on test set')
    print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,B_new,True,gamma_new,beta_new,last_mean,last_var))


    print('Training with nine layers with batch normalisation')
    print('------------------------------------------------ \n')
    W,B, gamma, beta = initialise([50,30,20,20,10,10,10,10,10],True,None,True)
    lamda = 0.005
    n_batch = 100
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    n_epochs = 20
    W_new, B_new, gamma_new, beta_new, last_mean, last_var, nine_layers_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda,True,gamma,beta,0.9),
        'W','B','gamma','beta','mean','var','training_log')
    print('Training complete, performing inference on test set')
    print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,B_new,True,gamma_new,beta_new,last_mean,last_var))
    

    #coarse hyperparameter search for lamda
    print('\n Coarse search for lamda')
    print('------------------------------------------------ \n')
    
    coarse_l = np.linspace(-5, -1, num=10)
    results = pd.DataFrame(index=coarse_l, columns = ['average valid accuracy'])
    n_batch, n_epochs = 100, 20
    #for 2 cycles
    #print(trainXall.shape[1])
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    #each epoch has 450 steps, to get 2 cycles: 4*5*450/450 = 20 epochs
    n_epochs = 20
    W_new, B_new, gamma_new, beta_new, last_mean, last_var, three_layers_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda,True,gamma,beta,0.9),
        'W','B','gamma','beta','mean','var','training_log')
    for power in coarse_l:
        lamda_test = 10**power
        W,B, gamma, beta= initialise([50,50,10],True,None,True)
        W_new, B_new, gamma_new, beta_new, last_mean, last_var, three_layers_bn = dictget(
            MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_test,True,gamma,beta,0.9),
            'W','B','gamma','beta','mean','var','training_log')
        curr_accuracy = ComputeAccuracy(validXall,validyall,W_new,B_new,True,gamma_new,beta_new,last_mean,last_var)
        results.loc[power,'average valid accuracy'] = curr_accuracy
    results.to_csv('coarse_search.csv')

    results = pd.read_csv('coarse_search.csv')
    results_top = results.sort_values('average valid accuracy', ascending=False).head(3)
    print(results_top.iloc[:,0])

    print('\n Fine search for lamda')
    print('------------------------------------------------ \n')

    fine_l = np.linspace(results_top.iloc[:,0].max(),results_top.iloc[:,0].min(),20)
    print(fine_l)
    fine_results = pd.DataFrame(index=fine_l, columns = ['average valid accuracy'])
    n_batch, n_epochs = 100, 20
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    for power in fine_l:
        lamda_test = 10**power
        W,B, gamma, beta= initialise([50,50,10],True,None,True)
        W_new, B_new, gamma_new, beta_new, last_mean, last_var, three_layers_bn = dictget(
            MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_test,True,gamma,beta,0.9),
            'W','B','gamma','beta','mean','var','training_log')
        curr_accuracy = ComputeAccuracy(validXall,validyall,W_new,B_new,True,gamma_new,beta_new,last_mean,last_var)
        fine_results.loc[power,'average test accuracy'] = curr_accuracy
    fine_results.to_csv('fine_search.csv')
    print(fine_results)


    #final performance with three layers
    fine_results = pd.read_csv('fine_search.csv')
    fine_results_top = fine_results.sort_values('average valid accuracy', ascending=False).head(3)
    print(fine_results_top)
    lamda_final = 10**fine_results_top.iloc[0,0]
    n_batch, n_epochs = 100, 30
    #for 3 cycles
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    W,B, gamma, beta= initialise([50,50,10],True,None,True)
    W_new, B_new, gamma_new, beta_new, last_mean, last_var, three_layers_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_final,True,gamma,beta,0.9),
        'W','B','gamma','beta','mean','var','training_log')
    final_accuracy = ComputeAccuracy(testX,testy,W_new,B_new,True,gamma_new,beta_new,last_mean,last_var)

    print('Model achieves test accuracy of: ', final_accuracy)



    #initialisation testing

    print('\n Initialisation without BN')
    print('------------------------------------------------ \n')

    lamda_final = 0.005
    n_batch, n_epochs = 100, 20
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    W,B = initialise([50,50,10],False,1e-1,False)
    W_new, B_new= dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_final),
        'W','B')
    final_accuracy = ComputeAccuracy(testX,testy,W_new,B_new,False)

    print('Model achieves test accuracy of: ', final_accuracy)

    W,B = initialise([50,50,10],False,1e-3,False)
    W_new, B_new = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_final),
        'W','B')
    final_accuracy = ComputeAccuracy(testX,testy,W_new,B_new,False)

    print('Model achieves test accuracy of: ', final_accuracy)


    W,B = initialise([50,50,10],False,1e-4,False)
    W_new, B_new = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_final),
        'W','B')
    final_accuracy = ComputeAccuracy(testX,testy,W_new,B_new,False)

    print('Model achieves test accuracy of: ', final_accuracy)

    print('\n Initialisation with BN')
    print('------------------------------------------------ \n')

    lamda_final = 0.005
    n_batch, n_epochs = 100, 20
    cyclical_1 = cyclical_eta(5*45000/n_batch,1e-5,1e-1)
    W,B, gamma, beta= initialise([50,50,10],False,1e-1,True)
    W_new, B_new, gamma_new, beta_new, last_mean, last_var, three_layers_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_final,True,gamma,beta,0.9),
        'W','B','gamma','beta','mean','var','training_log')
    final_accuracy = ComputeAccuracy(testX,testy,W_new,B_new,True,gamma_new,beta_new,last_mean,last_var)

    print('Model achieves test accuracy of: ', final_accuracy)

    W,B, gamma, beta= initialise([50,50,10],False,1e-3,True)
    W_new, B_new, gamma_new, beta_new, last_mean, last_var, three_layers_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_final,True,gamma,beta,0.9),
        'W','B','gamma','beta','mean','var','training_log')
    final_accuracy = ComputeAccuracy(testX,testy,W_new,B_new,True,gamma_new,beta_new,last_mean,last_var)

    print('Model achieves test accuracy of: ', final_accuracy)


    W,B, gamma, beta= initialise([50,50,10],False,1e-4,True)
    W_new, B_new, gamma_new, beta_new, last_mean, last_var, three_layers_bn = dictget(
        MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_1,n_epochs,W,B,lamda_final,True,gamma,beta,0.9),
        'W','B','gamma','beta','mean','var','training_log')
    final_accuracy = ComputeAccuracy(testX,testy,W_new,B_new,True,gamma_new,beta_new,last_mean,last_var)

    print('Model achieves test accuracy of: ', final_accuracy)


    

if __name__ == '__main__':
    main()