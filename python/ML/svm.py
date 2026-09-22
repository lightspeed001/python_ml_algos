# Support Vector Machine
from sklearn.svm import SVC
from sklearn.datasets import make_circles

X-svm, y_svm = make_circles(noise=0.1, factor=0.5, random_state=0)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_svm, y_svm)
print("support vectors: ", svm.support_vectors_.shape)
