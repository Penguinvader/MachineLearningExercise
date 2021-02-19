import matplotlib.pyplot as plt
import numpy as np
from numpy.core.function_base import linspace
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc, confusion_matrix, accuracy_score, plot_roc_curve, roc_auc_score
from sklearn.multiclass import OneVsOneClassifier
import urllib.request

with urllib.request.urlopen('https://raw.githubusercontent.com/Penguinvader/MLFiles/main/PhishingData.csv') as f:
    numpy_array = np.loadtxt(f, delimiter=",")

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=2, max_iter=1500)

data = numpy_array[:, :-1]
target = numpy_array[:, -1]
target = np.array([abs(i)+i for i in target])
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=777)


clf.fit(X_train, y_train)
logreg_clf = LogisticRegression(solver='liblinear')
logreg_clf.fit(X_train,y_train);

logreg_conf = plot_confusion_matrix(logreg_clf, X_test, y_test, display_labels=['Phishy/Suspicious', 'Legitimate'], normalize='true')
mlp_conf = plot_confusion_matrix(clf, X_test, y_test, display_labels=['Phishy/Suspicious', 'Legitimate'], normalize='true')
logreg_conf.ax_.set_title('LogReg Classifier')
mlp_conf.ax_.set_title('MLP Classifier')

logreg_roc = plot_roc_curve(logreg_clf, X_test, y_test);
mlp_roc = plot_roc_curve(clf, X_test, y_test, ax = logreg_roc.ax_);
logreg_roc.ax_.set_title('Roc curves')

plt.show()