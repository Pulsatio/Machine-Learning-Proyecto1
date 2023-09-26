
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pywt
import pywt.data
import numpy as np
import os
from sklearn.utils import shuffle
import random
import pandas as pd
from sklearn.model_selection import KFold

random.seed(7)




"""
def get_feature(picture, cortes = 3):
   LL = picture
   for i in range(cortes):
      LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
   return LL.flatten()

list_image = os.listdir("./images");

total_data = []
x_data_aux = []
y_data_aux = []
total_size = 0
for file in list_image:
	total_size += 1
	tipo = int(file[:3])
	imagen = imread("./images/" + file)
	img_res = transform.resize(imagen, (100, 100, 3))
	img_encoding = get_feature(img_res, 6)
	total_data.append((img_encoding,tipo))

random.shuffle(total_data)
for e in total_data:
	x_data_aux.append(e[0])
	y_data_aux.append(e[1])
for i in range(len(x_data_aux[0])):
	maxi = 0
	for j in range(len(x_data_aux)):
		maxi = max(maxi,x_data_aux[j][i])
	for j in range(len(x_data_aux)):
		x_data_aux[j][i] /= maxi

x_data = np.array(x_data_aux)
y_data = np.array(y_data_aux)



"""

dataset = pd.read_csv('vc_mariposas_sin_fondo_250_4.csv')
X = np.array(dataset['X'].apply(eval))
X = np.array([np.array(x) for x in X])
Y = dataset['Y'].values
Y = np.array([i - 1 for i in Y])

# Mezcla los datos
X, Y = shuffle(X, Y)#random_state

min_vals = X.min(axis=0)
max_vals = X.max(axis=0)
X = (X - min_vals) / (max_vals - min_vals)

total_size = len(X)

x_train = X[:(total_size*70)//100]
x_validation = X[(total_size*7)//10:(total_size*85)//100]
x_testing = X[(total_size*85)//100:]

y_train = Y[:(total_size*70)//100]
y_validation = Y[(total_size*7)//10:(total_size*85)//100]
y_testing = Y[(total_size*85)//100:]

def distance_minkowski(a,b,p):
	
	return np.power(np.sum( abs(a - b) ** p), 1/p )

class KNN:
	p = -1
	k = -1
	def train(self,_x_train,_y_train):
		self.x_train = _x_train
		self.y_train = _y_train

	def predict(self,x):
		size = len(self.x_train)
		result = []

		for i in range(size):
			distance = distance_minkowski(x,self.x_train[i],self.p)
			result.append((distance, y_train[i]))
	
		result.sort()

		result = result[:self.k]

		datos = {}
		maxi, maxi_tipo = 0,-1
		
		for par in result:
			if par[1] not in datos:
				datos[par[1]] = 0
			datos[par[1]] += 1
			if maxi < datos[par[1]]:
				maxi = datos[par[1]]
				maxi_tipo = par[1]	
		return maxi_tipo	
		

	def score(self,x_testing,y_testing):
		size = len(x_testing)
		correct = 0
		for i in range(size):
			if self.predict(x_testing[i])  == y_testing[i]:
				correct += 1
		correct /= size
		return correct * 100

	def generate_predict(self,x_testing):
		result = []
		for i in range(len(x_testing)):
			result.append(self.predict(x_testing[i]))
		return result

	def __init__(self,_x_train,_y_train, _k = -1, _p  = -1):
		self.x_train = _x_train
		self.y_train = _y_train

		if _k + _p != -2:
			self.k = _k 
			self.p = _p
			return

		max_efectivo,best_k, best_p = 0,0,0	

		for k in range(2,40):
			for p in range(1,5):
				self.k = k
				self.p = p
				resultado = self.score(x_testing,y_testing)
				print(k,p,resultado)
				if max_efectivo < resultado:
					max_efectivo = resultado
					best_k = k
					best_p = p

		self.k = best_k
		self.p = best_p
	
modelo = KNN(x_train,y_train) # se coloco luego de generar los K y P más precisos

print("K = ", modelo.k)
print("P = ", modelo.p)


k_folds = KFold(n_splits = 5)

scores = []



for train_index, test_index in k_folds.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    modelo.train(X_train, y_train)

    # Evalúa el modelo en el conjunto de prueba
    score = modelo.score(X_test, y_test)
    scores.append(score)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", sum(scores)/len(scores))
print("Number of CV Scores used in Average: ", len(scores))



from sklearn import metrics
import seaborn as sns
def make_confusion_matrix(Y_real, Y_pred, title="Confusion Matrix"):
    cm = metrics.confusion_matrix(Y_real, Y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    dataframe = pd.DataFrame(cmn)

    sns.heatmap(dataframe, annot=True, cbar=False, cmap="Blues")
    plt.title(title), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()

    eff = np.nanmean(np.diagonal(cmn))
    return eff


# Importa la biblioteca sklearn
from sklearn.metrics import confusion_matrix

# Calcula la matriz de confusión
y_pred = modelo.generate_predict(x_testing)
conf_matrix = confusion_matrix(y_testing, Y_pred)

# Imprime la matriz de confusión
print(conf_matrix)