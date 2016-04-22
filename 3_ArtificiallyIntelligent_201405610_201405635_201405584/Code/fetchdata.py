
from nltk.corpus import stopwords
from stemming.porter2 import stem
stop=stopwords.words('english')
fw=open("20ng-test-stemmed.txt","w")
with open("./datasets/20ng-test-all-terms.txt","r") as fo :
	for line in fo :
		c=line.split('\t')
		noshort=""
		words=c[1].split(' ')
		for word in words :
			word=word.replace('\n','')
			if len(word) >= 3 and word not in stop :
				noshort=str(noshort)+stem(word)+" "
		fw.write(c[0]+"\t"+str(noshort)+"\n")
fw.close() 


fw=open("20ng-train-stemmed.txt","w")
with open("./datasets/20ng-train-all-terms.txt","r") as fo :
	for line in fo :
		c=line.split('\t')
		noshort=""
		words=c[1].split(' ')
		for word in words :
			word=word.replace('\n','')
			if len(word) >= 3 and word not in stop :
				noshort=str(noshort)+stem(word)+" "
		fw.write(c[0]+"\t"+str(noshort)+"\n")
fw.close() 
