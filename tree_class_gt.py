import	argparse, sys
import	numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO

# for RF
from sklearn.ensemble import ExtraTreesClassifier

# for SVM
from sklearn import svm
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import *
from sklearn.svm import SVR

# for MLP
from sknn.mlp import Classifier, Layer
from sknn import mlp

def	is_in(a, L):
	for i in L:
		if a == i:
			return 1
	return -1

def	read_data(FF, LabCol):
	f = open(FF, "r")
	x = f.readlines()
	f.close()

	if LabCol == -1:
		xx = x[0].split('\t')
		LC = len(xx) - 1
	else:
		LC = LabCol
	D = []
	L = []
	# The number of classes
	nL = 0
	for i in x:
		xx = i.split('\t')
		tmp = []
		for s in range(len(xx) - 1):
			try:
				tmp.append(float(xx[s]))
			except:
				tmp.append(0.0)
		D.append(tmp)
		lab = int(xx[LC])
		#lab = int(xx[len(xx)-1])
		if is_in(lab, L) == -1:
			nL = nL + 1
		L.append(lab)
	return [D, L, nL]

def	read_var_names(FF):
	f = open(FF, "r")
	x = f.readlines()
	f.close()

	V = []
	for i in x:
		V.append(i[0:len(i)-1])
	return V

def	save_relevance(FF, Vars, Relevance):
	f = open(FF, "w")
	for v in range(len(Vars)):
	#for v in range(len(Vars)-1):
		f.write(Vars[v] + '\t' + str(Relevance[v]) + '\n')
	f.close()

def	save_confusion_matrix(FF, CM, nL):
	f = open(FF, "w")
	for i in range(0, nL):
		for j in range(0, nL):
			f.write(str(i) + '\t' + str(j) + '\t' + str(CM[i][j]) + '\n')
	f.close()

def	pos(a, V):
	ln = len(V)
	for i in range(ln):
		if a == V[i]:
			return i
	return -1

def	save_pdfj(PDF_J, Vars, FF):
	Pos = {}
	for i in Vars:
		p = pos(i, Vars)
		Pos[i] = p
	f = open(FF, "w")
	for i in Vars:
		for j in Vars:
			f.write(str(Pos[i]) + "\t" + str(Pos[j]) + "\t" + i + "\t" + j + "\t" + str(PDF_J[i,j]) + "\n")
		f.write("\n")
	f.close()

def	save_pdfjx(PDF_J, Vars, FF):
	Pos = {}
	for i in Vars:
		p = pos(i, Vars)
		Pos[i] = p
	f = open(FF, "w")
	for i in Vars:
		for j in Vars:
			for k in Vars:
				f.write(str(Pos[i]) + "\t" + str(Pos[j]) + "\t" + str(Pos[k]) + "\t" + i + "\t" + j + "\t" + k + "\t" + str(PDF_J[i,j,k]) + "\n")
		f.write("\n")
	f.close()

# To obtain the name of the variables n each tree. Obtained from:
# http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
def get_lineage(tree, feature_names):
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]     

     def recurse(left, right, child, lineage=None):          
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     VV = []
     for child in idx:
          for node in recurse(left, right, child):
               #print "n = ", node
               try:
                    att = node[3]
                    if is_in(att, VV) == -1:
                            VV.append(node[3])
               except:
                    bb = 1
     return VV

def	save_predict(FF, Label, Res):
	f = open(FF, "w")
	for i, L in enumerate(Label):
		f.write(str(L) + "\t" + str(Res[i]) + "\n")
	f.close()

parser = argparse.ArgumentParser()
parser.add_argument('-i', action = "store", dest = "i", help = "The input file")
parser.add_argument('-o', action = "store", dest = "o", help = "The output file")
parser.add_argument('-op', action = "store", dest = "op", help = "The output file containing the joinf PDF of the variables")
parser.add_argument('-op2', action = "store", dest = "op2", help = "The output file containing the joinf PDF of the variables, three vars")
parser.add_argument('-v', action = "store", dest = "v", help = "The file containing the variable names")
parser.add_argument('-nL', action = "store", dest = "nL", help = "The number of classes")
parser.add_argument('-md', action = "store", dest = "md", help = "Max depth of the tree")
parser.add_argument('-cm', action = "store", dest = "cm", help = "The output file in which the confusion matrix is saved")
parser.add_argument('-rel', action = "store", dest = "rel", help = "The output file in which the relevance of each variable is stored")
parser.add_argument('-relrf', action = "store", dest = "relrf", help = "The output file in which the relevance of each variable is stored (RF)")
parser.add_argument('-relsvm', action = "store", dest = "relsvm", help = "The output file in which the relevance of each variable is stored (SVM)")
parser.add_argument('-cmrel', action = "store", dest = "cmsvm", help = "The output file in which the confusion matrix is saved (SVM)")
parser.add_argument('-cmmlp', action = "store", dest = "cmmlp", help = "The output file in which the confusion matrix is saved (MLP)")
parser.add_argument('-pr', action = "store", dest = "pr", help = "The output file containing the predicted class")
parser.add_argument('-mt', action = "store", dest = "mt", help = "The criteria for removing entropy (entropy / gini)")
parser.add_argument('-LC', help = "The column in which the label is located")

args = parser.parse_args()

LabCol = -1
if args.LC:
	LabCol = int(args.LC)
else:
	LabCol = -1

md = int(args.md)
mt = args.mt

clf = tree.DecisionTreeClassifier(criterion = mt, max_depth = md)

[Data, Label, nL] = read_data(args.i, LabCol)


Vars = read_var_names(args.v)

#nL = int(args.nL) + 1

clf = clf.fit(Data, Label)

# Confusion matrix

CM = [None] * nL
for i in range(nL):
	CM[i] = [0.0] * nL

Err = 0.0
Res = clf.predict(Data)
for r in range(len(Res)):
	cl = Label[r]
	pred = Res[r]
	CM[cl][pred] = CM[cl][pred] + 1.0
	#Err = Err + abs(cl - pred)
	if cl != pred:
		Err = Err + 1.0

save_predict(args.pr, Label, Res)

Err = Err / float(len(Res))



with open(args.o, 'w') as f:
	f = tree.export_graphviz(clf, out_file=f)

save_relevance(args.rel, Vars, clf.feature_importances_)
#save_confusion_matrix(args.cm, CM, nL)

forest = ExtraTreesClassifier(n_estimators=1000, random_state = 0, criterion = mt, max_depth = md)
#forest = ExtraTreesClassifier(n_estimators=1000, random_state = 0, criterion = "gini", max_depth = md)

forest.fit(Data, Label)

# How often do the attributes are selected together (in the same tree)
lv = len(Vars)

Relevance = [0.0] * lv
for tree in forest.estimators_:
	for i in range(len(tree.feature_importances_)):
		Relevance[i] = Relevance[i] + tree.feature_importances_[i]
	LV = get_lineage(tree, Vars)
	print "LV = ", LV
save_relevance(args.relrf, Vars, Relevance)

n_feat = len(Vars)
n_targets = nL

print "SVM"
CM_SVM = [None] * (nL + 1)
for i in range(nL + 1):
	CM_SVM[i] = [0.0] * (nL + 1)

clf_svm = svm.LinearSVC()
#clf_svm = svm.SVC(decision_function_shape='ovo')
clf_svm.fit(Data, Label)
ResSVM = clf_svm.predict(Data)
ErrSVM = 0.0
for r in range(len(ResSVM)):
	cl = Label[r]
	pred = ResSVM[r]
	CM_SVM[cl][pred] = CM_SVM[cl][pred] + 1.0
	#Err = Err + abs(cl - pred)
	if cl != pred:
		ErrSVM = ErrSVM + 1.0

ErrSVM = ErrSVM / float(len(ResSVM))

