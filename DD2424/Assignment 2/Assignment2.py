import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def initialise():
    K = 10
    m = 50
    d = 3072
    W = []
    W.append(np.random.normal(0, 1/np.sqrt(d), size = (m,d)))
    W.append(np.random.normal(0, 1/np.sqrt(m), size = (K,m)))
    b = []
    b.append(np.zeros((m,1)))
    b.append(np.zeros((K,1)))
    return W,b

def SoftMax(s):
	return np.exp(s)/np.sum(np.exp(s),axis=0)

def relu(s):
	return np.maximum(0,s)

def EvaluateClassifier(X,W,b):
    h = relu(np.matmul(W[0],X)+b[0])
    return (h, SoftMax(np.matmul(W[1],h)+b[1]))

def ComputeCost(X,Y,W,b,lamda):
    p = EvaluateClassifier(X,W,b)[1]
    return (1/X.shape[1])*np.sum(-np.log((np.sum(Y*p, axis=0))))+lamda*np.sum(np.square(W[0]))+lamda*np.sum(np.square(W[1]))
    #EvaluateClassifier
	#find cross entropy loss
    #and then add L2 term

def ComputeLoss(X,Y,W,b,lamda):
    p = EvaluateClassifier(X,W,b)[1]
    return (1/X.shape[1])*np.sum(-np.log((np.sum(Y*p, axis=0))))

def ComputeAccuracy(X,y,W,b):
    p = EvaluateClassifier(X,W,b)[1]
    predictions = np.argmax(p, axis=0)
    return (y==predictions).sum()/len(y)
    #EvaluateClassifier
    #argmax along axis 0
    #divide sum of matching prediction over length to get accuracy


def ComputeGrads(X,Y,h,P,W,b,lamda):
    #Add gradient wrt b
    #Add gradient wrt W and add gradient for regularisation term
    #divide by number of entries

	#number of entries in batch, i.e cols in X
	n = X.shape[1]
	#shape of g, Y, P: (10, n) (each column is one sample's probabilty/one-hot-encoding/loss for each class)
	g = -(Y-P)

	#shape of grad wrt b for one sample: 10x1
	#so we need to post multiply to g, a nx1 matrix of 1s, sum up all samples for each value in b
	grad_b_2 = 1/n*np.matmul(g,np.ones((n,1), dtype="double"))
	#shape of grad wrt W for one sample: 10x3072
	grad_W_2 = 1/n*np.matmul(g,h.T)

	g = np.matmul(W[1].T,g)
	h[h>np.double()] = np.double(1)
	g = np.multiply(g,h)

	grad_b_1 = 1/n*np.matmul(g,np.ones((n,1), dtype="double"))
	grad_W_1 = 1/n*np.matmul(g,X.T)

	grad_W_1 += 2*lamda*W[0]
	grad_W_2 += 2*lamda*W[1]

	return [grad_W_1, grad_W_2], [grad_b_1, grad_b_2]

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """

	grad_W = [np.zeros(x.shape) for x in W]
	grad_b = [np.zeros(x.shape) for x in b]

	c = ComputeCost(X, Y, W, b, lamda)
	
	for j in range(len(b)):
		for i in range(len(b[j])):
			b_try = [np.array(x) for x in b]
			b_try[j][i] += h
			c2 = ComputeCost(X, Y, W, b_try, lamda)
			grad_b[j][i] = (c2-c) / h


	for k in range(len(W)):
		for j in range(W[k].shape[0]):
			for i in range(W[k].shape[1]):
				W_try = [np.array(x) for x in W]
				W_try[k][j][i] += h
				c2 = ComputeCost(X, Y, W_try, b, lamda)
				grad_W[k][j][i] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """

	grad_W = [np.zeros(x.shape) for x in W]
	grad_b = [np.zeros(x.shape) for x in b]
	
	for j in range(len(b)):
		for i in range(len(b[j])):
			b_try = [np.array(x) for x in b]
			b_try[j][i] -= h
			c1 = ComputeCost(X, Y, W, b_try, lamda)

			b_try = [np.array(x) for x in b]
			b_try[j][i] += h
			c2 = ComputeCost(X, Y, W, b_try, lamda)
			grad_b[j][i] = (c2-c1) / (2*h)
			

	for k in range(len(W)):
		for j in range(W[k].shape[0]):
			for i in range(W[k].shape[1]):
				W_try = [np.array(x) for x in W]
				W_try[k][j][i] -= h
				c1 = ComputeCost(X, Y, W_try, b, lamda)
				grad_W[k][j][i] = (c2-c1) / (2*h)
			
				W_try = [np.array(x) for x in W]
				W_try[k][j][i] += h
				c2 = ComputeCost(X, Y, W_try, b, lamda)
				grad_W[k][j][i] = (c2-c1) / (2*h)

	return [grad_W, grad_b]



def compareGrads(X, Y, W, b, lamda, h=1e-5):
	n = X.shape[1]
	print('Number of entries: ', n)
	print('Lambda: ', lamda, '\n')
	H, P = EvaluateClassifier(X,W,b)

	def findRelativeError(gradA, gradB):
		grad_shape = gradA.shape
		#print(np.abs(gradA-gradB))
		relative_error = np.divide(np.abs(gradA-gradB),np.maximum(np.finfo(np.float32).eps, np.abs(gradA)+np.abs(gradB)))
		print(f'Mean relative error: {np.mean(relative_error)}')
		return(np.allclose(relative_error, np.zeros(grad_shape,dtype="double"),rtol=0.0,atol=1e-3))
	
	grad_W ,grad_b = ComputeGrads(X, Y, H, P, W, b, lamda)
	
	#check with num
	print('Comparing with numerical method \n')
	grad_W_num ,grad_b_num = ComputeGradsNum(X, Y, P, W, b, lamda, h)
	print('Error for grad wrt W1: ', not findRelativeError(grad_W[0], grad_W_num[0]))
	print('Error for grad wrt W2: ', not findRelativeError(grad_W[1], grad_W_num[1]))
	print('Error for grad wrt b1: ', not findRelativeError(grad_b[0], grad_b_num[0]))
	print('Error for grad wrt b2: ', not findRelativeError(grad_b[1], grad_b_num[1]), '\n')

	print('Comparing with slow numerical method \n')
	grad_W_num_slow ,grad_b_num_slow = ComputeGradsNumSlow(X, Y, P, W, b, lamda, h)
	print('Error for grad wrt W1: ', not findRelativeError(grad_W[0], grad_W_num_slow[0]))
	print('Error for grad wrt W2: ', not findRelativeError(grad_W[1], grad_W_num_slow[1]))
	print('Error for grad wrt b1: ', not findRelativeError(grad_b[0], grad_b_num_slow[0]))
	print('Error for grad wrt b2: ', not findRelativeError(grad_b[1], grad_b_num_slow[1]), '\n')

	print('------------------------------------------------ \n')


def MiniBatchGD(X, Y, y, validX, validY, validy, n_batch, eta, n_epochs, W, b, lamda):
	Wstar = W.copy()
	bstar = b.copy()

	n = X.shape[1]

	t = 0

	training_data = pd.DataFrame(columns = ['train_loss','train_cost','train_acc','val_loss','val_cost','val_acc','eta'])
	for i in range(n_epochs):
		for j in range(n//n_batch):
			eta_t = eta(t)
			start = j*n_batch
			end = (j+1)*n_batch
			X_batch = X[:,start:end]
			Y_batch = Y[:,start:end]
			H,P = EvaluateClassifier(X_batch,Wstar,bstar)
			grad_W, grad_b = ComputeGrads(X_batch,Y_batch,H,P,Wstar,bstar,lamda)
			for k in range(len(Wstar)):
				Wstar[k] -= eta_t*grad_W[k]
				bstar[k] -= eta_t*grad_b[k]
			if (t%(n//n_batch*n_epochs//100)) == 0:
				train_loss = ComputeLoss(X,Y,Wstar,bstar,lamda)
				valid_loss = ComputeLoss(validX,validY,Wstar,bstar,lamda)
				training_data.loc[t,'eta'] = eta_t
				training_data.loc[t,'train_loss'] = train_loss
				training_data.loc[t,'train_cost'] = ComputeCost(X,Y,Wstar,bstar,lamda)
				training_data.loc[t,'train_acc'] = ComputeAccuracy(X,y,Wstar,bstar)
				training_data.loc[t,'val_loss'] = valid_loss
				training_data.loc[t,'val_cost'] = ComputeCost(validX,validY,Wstar,bstar,lamda)
				training_data.loc[t,'val_acc'] = ComputeAccuracy(validX,validy,Wstar,bstar)
			t += 1
		if i%10 == 0:
			print(f'On epoch {i} of {n_epochs}')
			print(f'Training loss: {train_loss}')
			print(f'Validation loss: {valid_loss}')
		

	ax = plt.subplot()
	ax.plot(training_data['train_loss'], label = 'training loss')
	ax.plot(training_data['val_loss'], label = 'validation loss')
	ax.set_xlabel('step')
	ax.set_ylabel('loss')
	ax.legend()
	ax.set_title(f'cross-entropy loss against step for lamda of {lamda}, batch size of {n_batch}, \n cyclical eta and {n_epochs} epochs')
	plt.savefig(f'{lamda}_lamda_{n_batch}_batches_cyclical_eta_{n_epochs}_epochs_loss.png', dpi=300)
	plt.close()
	ax = plt.subplot()
	ax.plot(training_data['train_cost'], label = 'training cost')
	ax.plot(training_data['val_cost'], label = 'validation cost')
	ax.set_xlabel('step')
	ax.set_ylabel('cost')
	ax.legend()
	ax.set_title(f'cross-entropy cost against step for lamda of {lamda}, batch size of {n_batch}, \n cylical eta and {n_epochs} epochs')
	plt.savefig(f'{lamda}_lamda_{n_batch}_batches_cylical_eta_{n_epochs}_epochs_cost.png', dpi=300)
	plt.close()
	ax = plt.subplot()
	ax.plot(training_data['eta'], label = 'eta')
	ax.set_xlabel('step')
	ax.set_ylabel('eta')
	ax.legend()
	ax.set_title(f'eta against step')
	plt.savefig(f'eta.png', dpi=300)
	plt.close()
	return [Wstar, bstar]


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
	trainX,trainY,trainy = load_batch('data_batch_1')
	validX,validY,validy = load_batch('data_batch_2')
	testX,testY,testy = load_batch('test_batch')
	trainX = normalise(trainX)
	validX = normalise(validX)
	testX = normalise(testX)
	W,b = initialise()

	#uncomment below to check gradients

	#test with 1st sample, lamda = 0
	compareGrads(trainX[:20,0].reshape(-1,1),trainY[:,0].reshape(-1,1), [W[0][:,:20], W[1]], b, 0, 1e-5)

	#test with 100 samples, lamda = 0
	compareGrads(trainX[:20,:100], trainY[:,:100], [W[0][:,:20], W[1]], b, 0, 1e-5)

	#test with 100 samples, lamda = 0.01
	compareGrads(trainX[:20,:100], trainY[:,:100], [W[0][:,:20], W[1]], b, 0.01, 1e-5)

	#test with 100 samples, lamda = 0.1
	compareGrads(trainX[:20,:100], trainY[:,:100], [W[0][:,:20], W[1]], b, 0.1, 1e-5)

	print('Testing by overfitting to training data')
	print('------------------------------------------------ \n')
	
	n_batch, eta, n_epochs, lamda = 10, lambda x:0.01 , 200, 0
	W_new, b_new = MiniBatchGD(trainX[:,:100],trainY[:,:100],trainy[:100],validX,validY,validy,n_batch,eta,n_epochs,W,b,lamda)
	print('Training complete, performing inference on test set')
	print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,b_new))
	

	print('\n Testing cyclical learning rate')
	print('------------------------------------------------ \n')
	cyclical_learning_rate_test = pd.DataFrame(index=np.arange(0,2000,1), columns = ['eta'])
	cyclical_test = cyclical_eta(500,1e-5,1e-1)
	for i in range(2000):
		cyclical_learning_rate_test.loc[i,'eta'] = cyclical_test(i)
	ax = plt.subplot()
	ax.plot(cyclical_learning_rate_test['eta'], label = 'eta')
	ax.set_xlabel('step')
	ax.set_ylabel('eta')
	plt.savefig(f'cyclical learning rate test.png', dpi=300)
	plt.close()

	#replicate figure 3
	n_batch = 100
	#trainX has 10000 samples now, means each epoch has 10000/100 = 100 steps
	#total of 10*100 = 1000 steps, means for 1 cycles, ns = (1000/2) = 500
	n_epochs = 10
	lamda = 0.01
	cyclical_test = cyclical_eta(500,1e-5,1e-1)
	W_new, b_new = MiniBatchGD(trainX,trainY,trainy,validX,validY,validy,n_batch,cyclical_test,n_epochs,W,b,lamda)
	print('Training complete, performing inference on test set')
	print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,b_new))

	#replicate figure 4
	W,b = initialise()
	n_batch = 100
	#trainX has 10000 samples now, means each epoch has 10000/100 = 100 steps
	#total of 48*100 = 4800 steps, means for 3 cycles, ns = (4800/6) = 800
	n_epochs = 48
	lamda = 0.01
	cyclical_test = cyclical_eta(800,1e-5,1e-1)
	W_new, b_new = MiniBatchGD(trainX,trainY,trainy,validX,validY,validy,n_batch,cyclical_test,n_epochs,W,b,lamda)
	print('Training complete, performing inference on test set')
	print('Model achieves test accuracy of: ', ComputeAccuracy(testX,testy,W_new,b_new))

	#coarse hyperparameter search for lamda
	print('\n Coarse search for lamda')
	print('------------------------------------------------ \n')
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
	
	coarse_l = np.linspace(-5, -1, num=10)
	results = pd.DataFrame(index=coarse_l, columns = ['average test accuracy'])
	n_batch, n_epochs = 100, 20
	#for 2 cycles
	#print(trainXall.shape[1])
	n_s = (trainXall.shape[1]/n_batch)*n_epochs // 4 
	#print(n_s)
	cyclical_rate = cyclical_eta(n_s,1e-5,1e-1)
	for power in coarse_l:
		lamda_test = 10**power
		W,b = initialise()
		W_new, b_new = MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_rate,n_epochs,W,b,lamda_test)
		curr_accuracy = ComputeAccuracy(validXall,validyall,W_new,b_new)
		results.loc[power,'average test accuracy'] = curr_accuracy
	results.to_csv('coarse_search.csv')

	results = pd.read_csv('coarse_search.csv')
	results_top = results.sort_values('average test accuracy', ascending=False).head(3)
	print(results_top.iloc[:,0])

	print('\n Fine search 1 for lamda')
	print('------------------------------------------------ \n')

	fine_l = np.linspace(results_top.iloc[:,0].max(),results_top.iloc[:,0].min(),10)
	print(fine_l)
	fine_results = pd.DataFrame(index=fine_l, columns = ['average test accuracy'])
	n_batch, n_epochs = 100, 40
	#for 2 cycles
	n_s = (trainXall.shape[1]/n_batch)*n_epochs // 4 
	cyclical_rate = cyclical_eta(n_s,1e-5,1e-1)
	for power in fine_l:
		lamda_test = 10**power
		W,b = initialise()
		W_new, b_new = MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_rate,n_epochs,W,b,lamda_test)
		curr_accuracy = ComputeAccuracy(validXall,validyall,W_new,b_new)
		fine_results.loc[power,'average test accuracy'] = curr_accuracy
	fine_results.to_csv('fine_search.csv')
	print(fine_results)

	print('\n Fine search 2 for lamda')
	print('------------------------------------------------ \n')

	fine_results = pd.read_csv('fine_search.csv')
	fine_results_top = fine_results.sort_values('average test accuracy', ascending=False).head(3)
	fine2_l = [fine_results_top.iloc[:,0].min() + (fine_results_top.iloc[:,0].max() - fine_results_top.iloc[:,0].min())*np.random.uniform() for i in range(20)]
	fine2_results = pd.DataFrame(index=fine2_l, columns = ['average test accuracy'])
	n_batch, n_epochs = 100, 40
	#for 2 cycles
	n_s = (trainXall.shape[1]/n_batch)*n_epochs // 4 
	cyclical_rate = cyclical_eta(n_s,1e-5,1e-1)
	for power in fine2_l:
		lamda_test = 10**power
		W,b = initialise()
		W_new, b_new = MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_rate,n_epochs,W,b,lamda_test)
		curr_accuracy = ComputeAccuracy(validXall,validyall,W_new,b_new)
		fine2_results.loc[power,'average test accuracy'] = curr_accuracy
	fine2_results.to_csv('fine2_search.csv')
	print(fine2_results)

	#final performance
	fine2_results = pd.read_csv('fine2_search.csv')
	fine2_results_top = fine2_results.sort_values('average test accuracy', ascending=False).head(3)
	lamda_final = 10**fine2_results_top.iloc[0,0]
	n_batch, n_epochs = 100, 80
	#for 4 cycles
	n_s = (trainXall.shape[1]/n_batch)*n_epochs // 8
	cyclical_rate = cyclical_eta(n_s,1e-5,1e-1)
	W,b = initialise()
	W_final, b_final = MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,cyclical_rate,n_epochs,W,b,lamda_final)
	final_accuracy = ComputeAccuracy(testX,testy,W_final,b_final)
	print('Model achieves test accuracy of: ', final_accuracy)
	

if __name__ == '__main__':
	main()