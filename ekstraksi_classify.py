import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from stop_words import get_stop_words
from fnc import top_feats_per_cluster, plot_tfidf_classfeats_h

#load pre processed data
chat = pd.read_csv("pre2.txt", sep="\n", header=None)
chat.columns = ['isi']

#additional stopword removal
stop_words = get_stop_words('indonesian')
stopword = stop_words

#tfidf
vect = TfidfVectorizer(analyzer='word', stop_words=stopword, max_df=0.50, min_df=2)
X = vect.fit_transform(chat.isi)

#k-means clustering
#init
features = vect.get_feature_names()
n_clusters = 3
clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=1)
labels = clf.fit_predict(X)

#plot 2d
X_dense = X.todense()
pca = PCA(n_components=2).fit(X_dense)
coords = pca.transform(X_dense)
label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
                "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
colors = [label_colors[i] for i in labels]
plt.scatter(coords[:, 0], coords[:, 1], c=colors)
centroids = clf.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
plt.show()

#plo graph
plot_tfidf_classfeats_h(top_feats_per_cluster(X, labels, features, 0.1, 25))