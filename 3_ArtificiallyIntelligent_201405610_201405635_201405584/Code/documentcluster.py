from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import numpy as np


dataset = fetch_20newsgroups(subset='all', categories=None,
                             shuffle=True, random_state=42)


labels=dataset.target

K=np.unique(labels).shape[0]

vectorizer=TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')

corpus=vectorizer.fit_transform(dataset.data)

svd = TruncatedSVD(12)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
corpus = lsa.fit_transform(corpus)

km = KMeans(n_clusters=K, init='k-means++', max_iter=1000, n_init=1)

km.fit(corpus)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(corpus, km.labels_, sample_size=1000))

print("Top terms per cluster:")
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(K):
    print("Cluster %d:" % i,end='')
    for ind in order_centroids[i, :20]:
        print(' %s' % terms[ind],end='')
    print()



