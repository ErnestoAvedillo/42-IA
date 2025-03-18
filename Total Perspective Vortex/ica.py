import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
np.random.seed(0)
num_rows = 3000
n_samples = 3000 
time = np.linspace(0, 8, num_rows)
t = np.linspace(0,10, n_samples)
# create signals sources
s1 = np.sin(3*t) # a sine wave
s2 = np.sign(np.cos(6*time)) # a square wave
s3 = signal.sawtooth(2 *t) # a sawtooth wave
# combine single sources to create a numpy matrix
S = np.c_[s1,s2,s3]

# add a bit of random noise to each value
S += 0.2 * np.random.normal(size = S.shape)

# create a mixing matrix A
A = np.array([[1, 1.5, 0.5], [2.5, 1.0, 2.0], [1.0, 0.5, 4.0]])
X = S.dot(A.T)

#plot the single sources and mixed signals
plt.figure(figsize =(26,12) )
colors = ['red', 'blue', 'orange']

plt.subplot(2,1,1)
plt.title('True Sources')
for color, series in zip(colors, S.T):
    plt.plot(series, color)
plt.subplot(2,1,2)
plt.title('Observations(mixed signal)')
for color, series in zip(colors, X.T):
    plt.plot(series, color)
plt.show()

from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow, BlockMatrix
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors, DenseMatrix, Matrix
from sklearn import datasets
# create the standardizer model for standardizing the dataset

X_rdd = scaler.parallelize(X).map(lambda x:Vectors.dense(x) )
scaler = StandardScaler(withMean = True, withStd = False).fit("iris_rdd")

X_sc = scaler.transform(X_rdd)


#create the IndexedRowMatrix from rdd
X_rm = IndexedRowMatrix(X_sc.zipWithIndex().map(lambda x: (x[1], x[0])))

# compute the svd factorization of the matrix. First the number of columns and second a boolean stating whether 
# to compute U or not. 
svd_o = X_rm.computeSVD(X_rm.numCols(), True)

# svd_o.V is of shape n * k not k * n(as in sklearn)

P_comps = svd_o.V.toArray().copy()
num_rows = X_rm.numRows()
# U is whitened and projected onto principal components subspace.

S = svd_o.s.toArray()
eig_vals = S**2
# change the ncomp to 3 for this tutorial
#n_comp  = np.argmax(np.cumsum(eig_vals)/eig_vals.sum() > 0.95)+1
n_comp = 3
U = svd_o.U.rows.map(lambda x:(x.index, (np.sqrt(num_rows-1)*x.vector).tolist()[0:n_comp]))
# K is our transformation matrix to obtain projection on PC's subspace
K = (U/S).T[:n_comp]

input("Press enter to continue")
np.random.seed(100)
U1 = np.random.uniform(-1, 1, 1000)
U2 = np.random.uniform(-1, 1, 1000)

G1 = np.random.randn(1000)
G2 = np.random.randn(1000)

fig = plt.figure()

ax1 = fig.add_subplot(121, aspect = "equal")
ax1.scatter(U1, U2, marker = ".")
ax1.set_title("Uniform")


ax2 = fig.add_subplot(122, aspect = "equal")
ax2.scatter(G1, G2, marker = ".")
ax2.set_title("Gaussian")


plt.show()

A = np.array([[1, 0], [1, 2]])

U_source = np.array([U1,U2])
U_mix = U_source.T.dot(A)

G_source = np.array([G1, G2])
G_mix = G_source.T.dot(A)

# plot of our dataset

fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("Mixed Uniform ")
ax1.scatter(U_mix[:, 0], U_mix[:,1], marker = ".")

ax2 = fig.add_subplot(122)
ax2.set_title("Mixed Gaussian ")
ax2.scatter(G_mix[:, 0], G_mix[:, 1], marker = ".")


plt.show()   

# PCA and whitening the dataset
from sklearn.decomposition import PCA 
U_pca = PCA(whiten=True).fit_transform(U_mix)
G_pca = PCA(whiten=True).fit_transform(G_mix)

# let's plot the uncorrelated columns from the datasets
fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("PCA Uniform ")
ax1.scatter(U_pca[:, 0], U_pca[:,1], marker = ".")

ax2 = fig.add_subplot(122)
ax2.set_title("PCA Gaussian ")
ax2.scatter(G_pca[:, 0], G_pca[:, 1], marker = ".")
plt.show()