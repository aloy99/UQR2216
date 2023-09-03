import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import copy
from sklearn.model_selection import train_test_split


np.random.seed(1) #ensure same seed per run

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
    K= 10
    d= 3072
    W = np.random.normal(0, 0.01, size = (K,d))
    b = np.random.normal(0, 0.01, size = (K,1))
    return (W,b)

def SoftMax(s):
	return np.exp(s)/np.sum(np.exp(s),axis=0)

def EvaluateClassifier(X,W,b):
    return SoftMax(np.matmul(W,X)+b)



def ComputeCost(X,Y,W,b,lamda):
    p = EvaluateClassifier(X,W,b)
    return (1/X.shape[1])*np.sum(-np.log((np.sum(Y*p, axis=0))))+lamda*np.sum(np.square(W))
    #EvaluateClassifier
	#find cross entropy loss
    #and then add L2 term

def ComputeLoss(X,Y,W,b,lamda):
    p = EvaluateClassifier(X,W,b)
    return (1/X.shape[1])*np.sum(-np.log((np.sum(Y*p, axis=0))))

def ComputeAccuracy(X,y,W,b):
    p = EvaluateClassifier(X,W,b)
    predictions = np.argmax(p, axis=0)
    return (y==predictions).sum()/len(y)
    #EvaluateClassifier
    #argmax along axis 0
    #divide sum of matching prediction over length to get accuracy


def ComputeGrads(X,Y,P,W,b,lamda):
    #Add gradient wrt b
    #Add gradient wrt W and add gradient for regularisation term
    #divide by number of entries

	#number of entries in batch, i.e cols in X
	n = X.shape[1]
	#shape of g, Y, P: (10, n) (each column is one sample's probabilty/one-hot-encoding/loss for each class)
	g = -(Y-P)

	#shape of grad wrt b for one sample: 10x1
	#so we need to post multiply to g, a nx1 matrix of 1s, sum up all samples for each value in b
	#grad_b = 1/n*np.matmul(g,np.ones((n,1), dtype=np.float64))
	grad_b = 1/n*np.matmul(g,np.ones((n,1), dtype="double"))
	#shape of grad wrt W for one sample: 10x3072
	grad_W = 1/n*np.matmul(g,X.T) + 2*lamda*W

	return [grad_W, grad_b]


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = ComputeCost(X, Y, W, b, lamda);
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]



def compareGrads(X, Y, W, b, lamda, h=1e-6):
	n = X.shape[1]
	print('Number of entries: ', n)
	print('Lambda: ', lamda, '\n')
	P = EvaluateClassifier(X,W,b)

	def findRelativeError(gradA, gradB):
		grad_shape = gradA.shape
		#print(np.abs(gradA-gradB))
		relative_error = np.divide(np.abs(gradA-gradB),np.maximum(np.finfo("double").eps, np.abs(gradA)+np.abs(gradB)))
		print(f'Mean relative error: {np.mean(relative_error)}')
		return(np.allclose(relative_error, np.zeros(grad_shape,dtype="double"),rtol=0.0,atol=1e-4))
	
	grad_W ,grad_b = ComputeGrads(X, Y, P, W, b, lamda)
	
	#check with num
	print('Comparing with numerical method \n')
	grad_W_num ,grad_b_num = ComputeGradsNum(X, Y, P, W, b, lamda, h)
	print('Error for grad wrt W: ', not findRelativeError(grad_W, grad_W_num))
	print('Error for grad wrt b: ', not findRelativeError(grad_b, grad_b_num), '\n')

	print('Comparing with slow numerical method \n')
	grad_W_num_slow ,grad_b_num_slow = ComputeGradsNumSlow(X, Y, P, W, b, lamda, h)
	print('Error for grad wrt W: ', not findRelativeError(grad_W, grad_W_num_slow))
	print('Error for grad wrt b: ', not findRelativeError(grad_b, grad_b_num_slow), '\n')

	print('------------------------------------------------ \n')

image_count = 0


def MiniBatchGD(X, Y, y, validX, validY, validy, n_batch, eta, n_epochs, W, b, lamda, decay=False):
	Wstar = W
	bstar = b

	eta_start = eta
	n = X.shape[1]

	training_data = pd.DataFrame(index=np.arange(0,n_epochs,1), columns = ['train_loss','train_cost','train_acc','val_loss','val_cost','val_acc'])
	for i in range(n_epochs):
		for j in range(n//n_batch):
			start = j*n_batch
			end = (j+1)*n_batch
			X_batch = X[:,start:end]
			Y_batch = Y[:,start:end]
			P = EvaluateClassifier(X_batch,Wstar,bstar)
			grad_W, grad_b = ComputeGrads(X_batch,Y_batch,P,Wstar,bstar,lamda)
			Wstar -= eta*grad_W
			bstar -= eta*grad_b
		train_loss = ComputeLoss(X,Y,Wstar,bstar,lamda)
		valid_loss = ComputeLoss(validX,validY,Wstar,bstar,lamda)
		if i%10 == 0:
			print(f'On epoch {i} of {n_epochs}')
			print(f'Training loss: {train_loss}')
			print(f'Validation loss: {valid_loss}')
		training_data.loc[i,'train_loss'] = train_loss
		training_data.loc[i,'train_cost'] = ComputeCost(X,Y,Wstar,bstar,lamda)
		training_data.loc[i,'train_acc'] = ComputeAccuracy(X,y,Wstar,bstar)
		training_data.loc[i,'val_loss'] = valid_loss
		training_data.loc[i,'val_cost'] = ComputeCost(validX,validY,Wstar,bstar,lamda)
		training_data.loc[i,'val_acc'] = ComputeAccuracy(validX,validy,Wstar,bstar)
		#if i>0 and decay and (training_data.loc[i,'val_acc']-training_data.loc[i,'val_acc'])/training_data.loc[i-1,'val_acc'] < 0.1:
		if (i%20) == 0 and decay:
			eta = 0.25*eta

	global image_count
	ax = plt.subplot()
	ax.plot(training_data['train_loss'], label = 'training loss')
	ax.plot(training_data['val_loss'], label = 'validation loss')
	ax.set_xlabel('epoch')
	ax.set_ylabel('loss')
	ax.legend()
	ax.set_title(f'cross-entropy loss against epoch for lamda of {lamda}, {n_batch} batch size, \n {eta_start} eta and {n_epochs} epochs')
	plt.savefig(f'{lamda}_lamda_{n_batch}_batches_{n_epochs}_epochs_{eta_start}_eta__loss_{image_count}.png', dpi=300)
	plt.close()
	ax = plt.subplot()
	ax.plot(training_data['train_cost'], label = 'training cost')
	ax.plot(training_data['val_cost'], label = 'validation cost')
	ax.set_xlabel('epoch')
	ax.set_ylabel('cost')
	ax.legend()
	ax.set_title(f'cross-entropy cost against epoch for lamda of {lamda}, {n_batch} batch size, \n {eta_start} eta and {n_epochs} epochs')
	plt.savefig(f'{lamda}_lamda_{n_batch}_batches_{n_epochs}_epochs_{eta_start}_eta_cost_{image_count}.png', dpi=300)
	plt.close()
	image_count += 1
	return [Wstar, bstar]

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


def image_flip(img_curr):
	img = copy.deepcopy(img_curr)
	for i in range(3):
		start = i*1024
		for j in range(32):
			start_j = start+j*32
			img[start_j:start_j+32] = np.flip(img[start_j:start_j+32])
	return img

def random_aug(data, prob, aug_func):
	new_data = copy.deepcopy(data)
	n = new_data.shape[1]
	index = np.random.choice(n, round(prob*n), replace=False)
	new_data[:,index] = np.apply_along_axis(aug_func, 0, new_data[:,index])
	return new_data


#1. use all datasets, assignment 2 code (done)
#2. data augmentation (done)
#3. grid search for lamda, eta, batch size
#4. decaying learning (done)

def main():
	trainX,trainY,trainy = load_batch('data_batch_1')
	validX,validY,validy = load_batch('data_batch_2')
	testX,testY,testy = load_batch('test_batch')

	#testing flip
	# fig, ax = plt.subplots(2)
	# ax[0].imshow(validX[:,1].astype(np.intc).reshape(3,32,32).transpose(1,2,0))
	# ax[1].imshow(image_flip(validX[:,1]).astype(np.intc).reshape(3,32,32).transpose(1,2,0))
	# plt.show()

	trainX = normalise(trainX)
	validX = normalise(validX)
	testX = normalise(testX)
	W,b = initialise()
	#P = EvaluateClassifier(trainX,W,b)

	#uncomment below to check gradients

	# #test with 1st sample, lamda = 0
	# compareGrads(trainX[:20,0].reshape(-1,1),trainY[:,0].reshape(-1,1), W[:,:20], b, 0, 1e-6)

	# #test with all samples, lamda = 0
	# compareGrads(trainX[:20,:], trainY, W[:,:20], b, 0, 1e-6)

	# #test with all samples, lamda = 0.01
	# compareGrads(trainX[:20,:], trainY, W[:,:20], b, 0.01, 1e-6)

	# #test with all samples, lamda = 0.1
	# compareGrads(trainX[:20,:], trainY, W[:,:20], b, 0.1, 1e-6)

	trainXall,trainYall,trainyall = load_batch('data_batch_1')
	for i in range(2,6):
		trainX_i, trainY_i, trainy_i = load_batch(f'data_batch_{i}')
		trainXall = np.concatenate((trainXall, trainX_i), axis = 1)
		trainYall = np.concatenate((trainYall, trainY_i), axis = 1)
		trainyall.extend(trainy_i)

	#randomly select validation set and remove from train set

	trainXall, validXall, trainYall, validYall, trainyall, validyall = train_test_split(trainXall.T, trainYall.T, trainyall, test_size=0.02, random_state=42)

	trainXall, validXall, trainYall, validYall = trainXall.T, validXall.T, trainYall.T, validYall.T

	#data augmentation, flipping
	trainXall_flipped = random_aug(trainXall, 0.1, image_flip)
	montage(trainXall_flipped.T,'trainxflip.jpg')

	trainXall = normalise(trainXall)
	trainXall_flipped = normalise(trainXall_flipped)
	validXall = normalise(validXall)
	testX = normalise(testX)
	print('\n Non-decaying lamda')
	print('------------------------------------------------ \n')
	n_batch, eta, n_epochs, lamda = 100, 0.001, 40, 0.1
	W_new, b_new = MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,eta,n_epochs,W,b,lamda,False)
	print('Training complete, performing inference on test set')
	print('Model achieves accuracy of: ', ComputeAccuracy(testX,testy,W_new,b_new))

	print('\n Decaying eta')
	print('------------------------------------------------ \n')
	W,b = initialise()
	W_new, b_new = MiniBatchGD(trainXall,trainYall,trainyall,validXall,validYall,validyall,n_batch,eta,n_epochs,W,b,lamda,True)
	print('Training complete, performing inference on test set')
	print('Model achieves accuracy of: ', ComputeAccuracy(testX,testy,W_new,b_new))

	print('\n Decaying eta and flipped')
	print('------------------------------------------------ \n')
	lamda = 0.05
	W,b = initialise()
	W_new, b_new = MiniBatchGD(trainXall_flipped,trainYall,trainyall,validXall,validYall,validyall,n_batch,eta,n_epochs,W,b,lamda,True)
	print('Training complete, performing inference on test set')
	print('Model achieves accuracy of: ', ComputeAccuracy(testX,testy,W_new,b_new))


	#coarse hyperparameter search for lamda
	print('\n Grid search for lamda, eta and batch size')
	print('------------------------------------------------ \n')
	lamda_power_range = np.linspace(-3, -2, num= 5)
	eta_range = np.linspace(-3, -1, num = 5)
	batch_size_range = np.linspace(20,120,5)

	lamdav, etav, batchv = np.meshgrid(lamda_power_range, eta_range, batch_size_range)
	lamdav, etav, batchv = lamdav.reshape(1,-1), etav.reshape(1, -1), batchv.astype('int').reshape(1, -1)

	epochs = 20
	# results = pd.DataFrame(columns = ['lamda', 'eta', 'batch size', 'average validation accuracy'])
	# for i in range(125):
	# 	lamda = 10**lamdav[0][i]
	# 	n_batch = batchv[0][i]
	# 	eta = 10**etav[0][i]
	# 	print(f'Training with lambda of {lamda}, batch size of {n_batch}, eta of {eta} and {epochs} epochs')
	# 	W,b = initialise()
	# 	W_new, b_new = MiniBatchGD(trainXall_flipped,trainYall,trainyall,validXall,validYall,validyall,n_batch,eta,epochs,W,b,lamda,True)
	# 	curr_accuracy = ComputeAccuracy(validXall,validyall,W_new,b_new)
	# 	results.loc[len(results.index)] = {'lamda':lamda, 'eta': eta, 'batch size': n_batch, 'average validation accuracy': curr_accuracy}
	# results.to_csv('coarse_search.csv')

	results = pd.read_csv('coarse_search.csv')
	results_top = results.sort_values('average validation accuracy', ascending=False).head(3)
	print(results_top.head)

	#best params:
	#lamda of 0.003162, eta of 0.003162, batch size of 45
	lamda = 0.003162
	eta = 0.003162
	n_batch = 45
	epochs = 100
	W,b = initialise()
	W_new, b_new = MiniBatchGD(trainXall_flipped,trainYall,trainyall,validXall,validYall,validyall,n_batch,eta,epochs,W,b,lamda,True)
	curr_accuracy = ComputeAccuracy(testX,testy,W_new,b_new)
	print(curr_accuracy)







	

if __name__ == '__main__':
	main()