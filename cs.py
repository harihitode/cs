# make sure you've got the following packages installed
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
from PIL import Image

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# read original image and downsize for speed
Xorig = spimg.imread('escher_waterfall.jpeg', flatten=True, mode='L') # read in grayscale
X = spimg.zoom(Xorig, 0.04)
ny,nx = X.shape

Image.fromarray(Xorig.astype(np.uint8)).save('original.jpg')
Image.fromarray(X.astype(np.uint8)).save('zoom.jpg')

# extract small sample of signal
k = int(round(nx * ny * 0.5)) # 50% sample

print("col: ", nx);
print("row: ", ny);

ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

# Y = X;
# Y.T.flat[ri] = 255.0;
# Image.fromarray(Y.astype(np.uint8)).save('sample.jpg')

# print(np.identity(4))

# ar = np.array([1.0, 0.0, 0.0, 0.0,
#                0.0, 1.0, 0.0, 0.0,
#                0.0, 0.0, 1.0, 0.0,
#                0.0, 0.0, 0.0, 1.0]);
# print(ar);
# print(np.identity(4))
# print(spfft.idct(ar, norm='ortho', axis=0))
# print(spfft.idct(np.identity(4), norm='ortho', axis=0))

b = X.T.flat[ri]
print(b)
# create dct matrix operator using kron (memory errors for large ny*nx)
A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
)
print(A.shape)
print(A)
A = A[ri,:] # same as phi times kron
print(A.shape)
print(A)


# # do L1 optimization
# vx = cvx.Variable(nx * ny)
# objective = cvx.Minimize(cvx.norm(vx, 1))
# constraints = [A*vx == b]
# prob = cvx.Problem(objective, constraints)
# result = prob.solve(verbose=True)
# Xat2 = np.array(vx.value).squeeze()

# # reconstruct signal
# Xat = Xat2.reshape(nx, ny).T # stack columns

# Xa = idct2(Xat)

# Image.fromarray(Xa.astype(np.uint8)).save('recons.jpg')

# # confirm solution
# if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
#     print('Warning: values at sample indices don\'t match original.')

# # create images of mask (for visualization)
# mask = np.zeros(X.shape)
# mask.T.flat[ri] = 255
# Xm = 255 * np.ones(X.shape)
# Xm.T.flat[ri] = X.T.flat[ri]

# Image.fromarray(Xm.astype(np.uint8)).save('sample.jpg')
