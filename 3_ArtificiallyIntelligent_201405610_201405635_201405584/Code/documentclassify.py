from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression

def classifiers(clf,classifierstring,data_train,data_test,classes_train,classes_test) :
	print classifierstring
	clf.fit(data_train,classes_train)

	predictions=clf.predict(data_test)

	print "Accuracy : "+ str(metrics.accuracy_score(classes_test,predictions)*float(100))

	print "Confusion Matrix : "

	print confusion_matrix(predictions,classes_test)
	
	return clf

def svmclassifiers(kernelstr,classifierstring,data_train,data_test,classes_train,classes_test) :
	print classifierstring

	svm=train_svm(data_train,classes_train,kernelstr)

	predictions=svm.predict(data_test)

	print "Accuracy : " +str(svm.score(data_test,classes_test)*float(100))

	print "Confusion Matrix : "

	print confusion_matrix(predictions,classes_test)

def train_svm(X, y,kernelstr):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0,kernel=kernelstr)
    svm.fit(X, y)
    return svm

def createTFIDFVector(traindocuments,testdocuments) :
	trainclasses=[d[0] for d in traindocuments]
	testclasses=[d[0] for d in testdocuments]
	traincorpus=[d[1] for d in traindocuments]
	testcorpus=[d[1] for d in testdocuments]
	vectorizer=TfidfVectorizer(min_df=1)
	corpustrainvector=vectorizer.fit_transform(traincorpus)
	corpustestvector=vectorizer.transform(testcorpus)
	return corpustrainvector,corpustestvector,trainclasses,testclasses




traindocuments=[]
with open("20ng-train-stemmed.txt","r") as fo :
	for line in fo :
		c=line.split('\t')
		traindocuments.append((c[0],c[1]))
testdocuments=[]
with open("20ng-test-stemmed.txt","r") as fo :
	for line in fo :
		c=line.split('\t')
		testdocuments.append((c[0],c[1]))

#TfIDF Vector
traindata,testdata,trainclasses,testclasses=createTFIDFVector(traindocuments,testdocuments)

#CrossValidation
data_train, data_test, classes_train, classes_test = train_test_split(traindata, trainclasses, test_size=0.2, random_state=42)

clf=MultinomialNB(alpha=.03)
clf=classifiers(clf,"Multinomial Naive Bayes Cross Validation",data_train,data_test,classes_train,classes_test)
clf=classifiers(clf,"Multinomial Naive Bayes Final Corpus Testing",traindata,testdata,trainclasses,testclasses)


svmclassifiers('rbf',"RBF SVM Cross Validation",data_train,data_test,classes_train,classes_test)
svmclassifiers('rbf',"RBF SVM Final Corpus Testing",traindata,testdata,trainclasses,testclasses)
svmclassifiers('linear',"Linear SVM Cross Validation",data_train,data_test,classes_train,classes_test)
svmclassifiers('linear',"Linear SVM Final Corpus Testing",traindata,testdata,trainclasses,testclasses)

clf=LogisticRegression(multi_class='multinomial',solver='lbfgs',random_state=42)
clf=classifiers(clf,"Maximum Entropy Cross Validation",data_train,data_test,classes_train,classes_test)
clf=classifiers(clf,"Maximum Entropy Final Corpus Testing",traindata,testdata,trainclasses,testclasses)


clf=BernoulliNB(alpha=.03)
clf=classifiers(clf,"Bernoulli Naive Bayes Cross Validation",data_train,data_test,classes_train,classes_test)
clf=classifiers(clf,"Bernoulli Naive Bayes Final Corpus Testing",traindata,testdata,trainclasses,testclasses)

clf=RandomForestClassifier(n_estimators=250)
clf=classifiers(clf,"Random Forest Cross Validation",data_train,data_test,classes_train,classes_test)
clf=classifiers(clf,"Random Forest Final Corpus Testing",traindata,testdata,trainclasses,testclasses)

clf=KNeighborsClassifier(n_neighbors=17)
clf=classifiers(clf,"KNN Cross Validation",data_train,data_test,classes_train,classes_test)
clf=classifiers(clf,"KNN Final Corpus Testing",traindata,testdata,trainclasses,testclasses)

clf=tree.DecisionTreeClassifier()
clf=classifiers(clf,"Decision Tree Cross Validation",data_train,data_test,classes_train,classes_test)
clf=classifiers(clf,"Decision Tree Final Corpus Testing",traindata,testdata,trainclasses,testclasses)







