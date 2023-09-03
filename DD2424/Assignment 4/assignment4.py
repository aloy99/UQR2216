import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import sys
import pickle

#for dict destructuring
dictget = lambda d, *k: [d[i] for i in k]

rng = np.random.default_rng(42) #ensure same seed per run

class RNN:
    def __init__(self,K,m=100,sig=0.01,eta=0.1,seq_length=25):
        self.K = K
        self.m = m
        self.sig = sig
        self.eta = eta
        self.seq_length = seq_length

        self.U = rng.normal(0, 1, size = (self.m,self.K))*self.sig
        self.W = rng.normal(0, 1, size = (self.m,self.m))*self.sig
        self.V = rng.normal(0, 1, size = (self.K,self.m))*self.sig

        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K,1))
    
def load_file(filename):
    with open(filename, 'r',encoding='utf-8') as x:
        text = x.read()
        return text
    
def build_dicts(text):
    unique_chars = set(text)
    #print('this is the set')
    #print(unique_chars)
    char_to_idx, idx_to_char = {}, {}
    for i,char in enumerate(unique_chars):
        char_to_idx[char] = i
        idx_to_char[i] = char
        i += 1
    return char_to_idx, idx_to_char

def encoder(text, char_to_idx):
    return np.eye(len(char_to_idx))[[char_to_idx[char] for char in text]].T

def decoder(output, idx_to_char):
    return "".join([idx_to_char[n] for n in np.argmax(output,axis=0)])

def softmax(s):
    #https://stats.stackexchange.com/questions/304758/softmax-overflow
    #deduct max of S from each value, mathematically same but avoids overflow
    max_s = np.max(s)
    return np.exp(s-max_s)/np.sum(np.exp(s-max_s),axis=0)

def synthesise_chars(rnn,h0,x0,n):
    x_t = copy.deepcopy(x0).reshape(-1,1)
    h_t = copy.deepcopy(h0).reshape(-1,1)
    Y = np.zeros((rnn.K,n))
    for i in range(n):
        a_t = np.matmul(rnn.W,h_t)+np.matmul(rnn.U,x_t)+rnn.b
        h_t = np.tanh(a_t)
        o_t = np.matmul(rnn.V,h_t)+rnn.c
        p_t = softmax(o_t)
        x_choice = rng.choice(rnn.K,p=p_t.reshape(-1))
        x_t = np.zeros(x_t.shape)
        x_t[x_choice] = 1
        Y[:,i] = x_t.reshape(-1)

    return Y

def forward_pass(rnn,h0,X,Y):
    #perform forward pass, same as synthesis but we store in
    #arrays of length of X (input seq)
    #compute loss for each 
    #store loss in array
    #store h
    #store a (we use with t+1)
    #store x at each t (we alr have it as an input though)
    X_len = X.shape[1]
    losses = np.zeros((1,X_len))
    h = np.zeros((rnn.m,X_len))
    a = np.zeros((rnn.m,X_len))
    Y_prob = np.zeros((rnn.K,X_len))

    for i in range(X_len):
        if i == 0:
            a[:,i] = (np.matmul(rnn.W,h0)+np.matmul(rnn.U,X[:,i].reshape(-1,1))+rnn.b).reshape(-1)
        else:
            a[:,i] = (np.matmul(rnn.W,h[:,i-1].reshape(-1,1))+np.matmul(rnn.U,X[:,i].reshape(-1,1))+rnn.b).reshape(-1)
        h[:,i] = np.tanh(a[:,i])
        #don't need to store o or p for back pass
        o_t = np.matmul(rnn.V,h[:,i].reshape(-1,1))+rnn.c
        p_t = softmax(o_t)
        #instead of sampling, store probability, and compute loss
        Y_prob[:,i] = p_t.reshape(-1)
    
    loss = compute_loss(Y,Y_prob)

    return h,a,loss,Y_prob

def compute_grads(rnn,h,a,P,Y,h0,X):

    grad_a = [] #shape: 1xm * mxm (diag from col of a) = 1xm
    grad_h = [] #shape: 1xK * Kxm -> 1xm
    g = -(Y-P).T #shape: len_seq, K, access grad for time t by row i.e [i] instead of [:,i]
    
    grad_h.append(np.matmul(g[-1],rnn.V))
    grad_a.append(np.matmul(grad_h[-1],np.diag((1-np.power(np.tanh(a[:,-1]),2)))))

    for i in reversed(range(P.shape[1]-1,)):
        #using g_t, and a_(t+1), which is the latest a from prev iteration
        grad_h.append(np.matmul(g[i],rnn.V)+np.matmul(grad_a[-1],rnn.W))
        grad_a.append(np.matmul(grad_h[-1],np.diag((1-np.power(np.tanh(a[:,i]),2)))))
        
    grad_a.reverse()
    grad_a = np.stack(grad_a) #shape of t x m 
    grad_h.reverse()

    grad_rnn = RNN(rnn.K)
    grad_rnn.V = np.matmul(g.T,h.T) 
    grad_rnn.W = np.matmul(grad_a.T,np.c_[h0,h[:,0:-1]].T) #add h0, remove final h, for offset of h vs g
    grad_rnn.U = np.matmul(grad_a.T,X.T)
    #for grad wrt b: just grad wrt a * 1 vector, bc coeff = 1 and power of 1 (similar to bias for prev assignments)
    grad_rnn.b = np.matmul(grad_a.T,np.ones((grad_a.shape[0],1), dtype="double"))
    #for grad wrt c: just grad wrt o * 1 vector
    grad_rnn.c = np.matmul(g.T,np.ones((g.shape[0],1), dtype="double"))
    return grad_rnn

def compute_loss(y,p):
    return np.sum(-np.log((np.sum(y*p, axis=0))))

def compute_grads_num_slow(rnn,h0,X,Y,h):

    grad_rnn = RNN(rnn.K)
    for param in ['U','W','V','b','c']: #we can't index an object's attributes easily in python vs matlab, so we use getattr and setattr
        grad = np.zeros(getattr(rnn,param).shape)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                rnn_try = copy.deepcopy(rnn)
                temp = copy.deepcopy(getattr(rnn_try,param))
                temp[i][j] -= h
                setattr(rnn_try,param,temp)
                c1 = compute_loss(Y,forward_pass(rnn_try,h0,X,Y)[3])

                rnn_try = copy.deepcopy(rnn)
                temp = copy.deepcopy(getattr(rnn_try,param))
                temp[i][j] += h
                setattr(rnn_try,param,temp)
                c2 = compute_loss(Y,forward_pass(rnn_try,h0,X,Y)[3])

                grad[i][j] = (c2-c1) / (2*h)

        setattr(grad_rnn,param,grad)

    return grad_rnn


def compareGrads(rnn,Y,h0,X,h_eps=1e-4):
    h,a,_,P = forward_pass(rnn,h0,X,Y)
    def findRelativeError(gradA, gradB):
        grad_shape = gradA.shape
        relative_error = np.divide(np.abs(gradA-gradB),np.maximum(np.finfo(np.float32).eps, np.abs(gradA)+np.abs(gradB)))
        print(f'Max relative error: {np.max(relative_error)}')
        return(np.allclose(relative_error, np.zeros(grad_shape,dtype="double"),rtol=0.0,atol=1e-3))
    
    grad_rnn_analytic = compute_grads(rnn,h,a,P,Y,h0,X)

    print('Comparing with slow numerical method \n')
    grad_rnn_num_slow = compute_grads_num_slow(rnn,h0,X,Y,h_eps)
    
    for i in ['U','W','V','b','c']:
        print(f'Error for grad wrt {i}: ', not findRelativeError(getattr(grad_rnn_analytic,i), getattr(grad_rnn_num_slow,i)))

    print('------------------------------------------------ \n')

def adagrad(param,m,g,eta):
    m = m + np.power(g,2)
    #print(m)
    param = param - eta/np.sqrt(m+np.finfo(np.double).eps) * g
    return param,m

def train_rnn(rnn,text,n_epochs,char_to_idx,idx_to_char):
    
    training_log = pd.DataFrame(columns = ['smoothed loss','output'])
    epoch = 0
    step = 0
    smooth_loss = None
    best_model = rnn
    best_loss = 1e7
    m_adagrad = {'U':0,
                 'W':0,
                 'V':0,
                 'b':0,
                 'c':0}
    e = 0
    
    while epoch < n_epochs:
        hprev = np.zeros((rnn.m,1))
        
        x = encoder(text[e:e+rnn.seq_length],char_to_idx)
        y = encoder(text[e+1:e+rnn.seq_length+1],char_to_idx)
        h,a,loss,P = forward_pass(rnn,hprev,x,y)

        if not smooth_loss:
            smooth_loss = loss
        else:
            smooth_loss = 0.999*smooth_loss + 0.001*loss
        
        if smooth_loss < best_loss:
            best_loss = smooth_loss
            best_model = copy.deepcopy(rnn)

        grads_rnn = compute_grads(rnn,h,a,P,y,hprev,x)

        for i in ['U','W','V','b','c']:
            grad_i = getattr(grads_rnn,i)
            grad_i = np.maximum(np.minimum(grad_i,5),-5)
            new_param, new_m = adagrad(getattr(rnn,i),m_adagrad[i],grad_i,rnn.eta)
            m_adagrad[i] = new_m
            setattr(rnn,i,new_param)

        hprev = h[:,-1] #get last hidden state
        e += rnn.seq_length

        if step%100 == 0:
            training_log.loc[step,'smoothed loss'] = smooth_loss
        if step%500 == 0:
            output = decoder(synthesise_chars(rnn,hprev,x[:,1],200),idx_to_char)
            training_log.loc[step,'output'] = output
            print(f'On step {step}')
            print(f'Smoothed loss: {smooth_loss}')
            #print(f'Sum of W: {np.sum(rnn.W)}')
        if step%10000 == 0:
            print(output)
        step += 1
        if e > len(text)-rnn.seq_length-1:
            e = 0
            epoch += 1
            print(f'On epoch {epoch}')
            hprev = np.zeros((rnn.m,1))

    return best_model, training_log


def main():
    goblet = load_file('goblet_book.txt')
    char_to_idx, idx_to_char = build_dicts(goblet)
    #print(idx_to_char)

    rnn_test = RNN(len(char_to_idx))

    #test synthesis
    output = synthesise_chars(rnn_test,np.zeros((100,1)),np.zeros((rnn_test.K,1)),10)

    #test decode
    samples = decoder(output,idx_to_char)
    print(samples)

    #test encode
    print(encoder(goblet[0:25],char_to_idx).shape)

    #test forward pass
    test_x = encoder(goblet[0:25],char_to_idx)
    test_y = encoder(goblet[1:26],char_to_idx)
    h,a,_,P = forward_pass(rnn_test,np.zeros((100,1)),test_x,test_y)

    #test compute grads

    grad_test =  compute_grads(rnn_test,h,a,P,test_y,np.zeros((100,1)),test_x)
    #for i in ['U','W','V','b','c']:
        #print(getattr(grad_test,i).shape)
    # grad_num_test = compute_grads_num_slow(rnn_test,np.zeros((100,1)),test_x,test_y,1e-5)
    # for i in ['U','W','V','b','c']:
    #     print(getattr(grad_num_test,i).shape)
    #wow its really slow
    
    #for i in ['U','W','V','b','c']:
        #print(getattr(grad_test,i).shape)
    rnn_test_grads = RNN(len(char_to_idx),5)
    compareGrads(rnn_test_grads,test_y,np.zeros((5,1)),test_x)

    rnn = RNN(len(char_to_idx),100)
    best_rnn, train_log = train_rnn(rnn,goblet,10,char_to_idx,idx_to_char)
    train_log.to_csv('train_log.csv')
    # Store data (serialize)
    with open('best_model.pickle', 'wb') as handle:
        pickle.dump(best_rnn, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #with open('best_model.pickle', 'rb') as file:
        #best_rnn = pickle.load(file)

    #print(best_rnn.W)

    final_output = synthesise_chars(best_rnn,np.zeros((100,1)),encoder('.',char_to_idx),1000)
    with open('final_output.txt','w') as file:
        file.write(decoder(final_output,idx_to_char))

    train_log = pd.read_csv('train_log.csv')
    print(train_log.head())
    ax = plt.subplot()
    ax.plot(train_log.iloc[:,0],train_log['smoothed loss'], label = 'smoothed loss')
    ax.set_xlabel('step')
    ax.set_ylabel('smoothed loss')
    ax.legend()
    ax.set_title(f'smoothed loss against step')
    plt.savefig(f'loss.png', dpi=300)
    plt.close()
    

if __name__ == '__main__':
    main()