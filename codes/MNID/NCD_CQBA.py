#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
from sentence_transformers import SentenceTransformer,util
from collections import Counter, defaultdict
from scipy.spatial import distance
import numpy as np


# # Read Data

# In[ ]:


f =open('data/train.label')
train_labels=f.readlines()
f = open('data/valid.label')
valid_labels = f.readlines()


# In[ ]:


f = open('data/valid.seq.in')
valid_data=f.readlines()


# In[ ]:


f = open('OOD_Data/preds_banking77_banking77_data.csv')
output_preds = f.readlines()


# In[ ]:


new_labels=set()
new_lab_list=[]
ood_data =[]
for preds,gold,data in zip(output_preds,valid_labels,valid_data):
    if preds =='OOD\n':
        new_labels.add(gold)
        new_lab_list.append(gold)
        ood_data.append(data)


# In[ ]:


f = open('data/actual_gold_dev_data.txt')
gold = f.readlines()


# In[ ]:


model = SentenceTransformer('all-MiniLM-L6-v2')
sen_embeddings = model.encode(ood_data, convert_to_tensor=True)


# # NCD

# In[ ]:


all_annotations=[]
all_labels=[]


# In[ ]:


from sklearn.cluster import KMeans
num_clusters = 2
# Define kmeans model
random.seed(1)
clustering_model = KMeans(n_clusters=num_clusters,random_state=0)
# Fit the embedding with kmeans clustering.
clustering_model.fit(sen_embeddings)
cluster_assignment = clustering_model.labels_
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(ood_data[sentence_id])
random.seed(0)
annotations=[]
for i in range(len(clustered_sentences)):
    annotations.extend(random.sample(clustered_sentences[i],2))
all_annotations.extend(annotations)
labels_for_annotation=[]
for data in annotations:
    idx= valid_data.index(data)
    labels_for_annotation.append(valid_labels[idx])
all_labels.extend(labels_for_annotation)
new_classes = len(set(labels_for_annotation)-set(train_labels))
print(f'No of new classes discovered:{new_classes}')


# In[ ]:


from sklearn.cluster import KMeans
num_clusters = 4
random.seed(1)
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters,random_state=0)
# Fit the embedding with kmeans clustering.
clustering_model.fit(sen_embeddings)
cluster_assignment = clustering_model.labels_
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(ood_data[sentence_id])
random.seed(0)
annotations=[]
for i in range(len(clustered_sentences)):
    common = set(all_annotations) & set(clustered_sentences[i])
    annotations.extend(random.sample(clustered_sentences[i],2))
all_annotations.extend(annotations)
labels_for_annotation=[]
for data in annotations:
    idx= valid_data.index(data)
    labels_for_annotation.append(valid_labels[idx])
all_labels.extend(labels_for_annotation)
new_classes = len(set(labels_for_annotation)-set(train_labels))
print(f'No of new classes discovered:{new_classes}')


# In[ ]:


from sklearn.cluster import KMeans
num_clusters = 8
random.seed(1)
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters,random_state=0)
# Fit the embedding with kmeans clustering.
clustering_model.fit(sen_embeddings)
cluster_assignment = clustering_model.labels_
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(ood_data[sentence_id])
random.seed(0)
annotations=[]
for i in range(len(clustered_sentences)):
    common = set(all_annotations) & set(clustered_sentences[i])
    annotations.extend(random.sample(clustered_sentences[i],2))
all_annotations.extend(annotations)
labels_for_annotation=[]
for data in annotations:
    idx= valid_data.index(data)
    labels_for_annotation.append(valid_labels[idx])
all_labels.extend(labels_for_annotation)
new_classes = len(set(labels_for_annotation)-set(train_labels))
print(f'No of new classes discovered:{new_classes}')


# In[ ]:


cluster_no={}
annotation_with_labels={}
from sklearn.cluster import KMeans
num_clusters = 16
random.seed(1)
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters,random_state=0)
# Fit the embedding with kmeans clustering.
clustering_model.fit(sen_embeddings)
cluster_assignment = clustering_model.labels_
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(ood_data[sentence_id])
random.seed(0)
annotations=[]
for i in range(len(clustered_sentences)):
    common = set(all_annotations) & set(clustered_sentences[i])
    #if len(common) <2:
    annotations.extend(random.sample(clustered_sentences[i],2))
all_annotations.extend(annotations)
labels_for_annotation=[]
for data in annotations:
    idx= valid_data.index(data)
    labels_for_annotation.append(valid_labels[idx])
    annotation_with_labels[data]=valid_labels[idx]
all_labels.extend(labels_for_annotation)
new_classes = len(set(labels_for_annotation)-set(train_labels))
print(f'No of new classes discovered:{new_classes}')


# In[ ]:


cluster_no={}
annotation_with_labels={}
from sklearn.cluster import KMeans
num_clusters = 32
random.seed(1)
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters,random_state=0)
# Fit the embedding with kmeans clustering.
clustering_model.fit(sen_embeddings)
cluster_assignment = clustering_model.labels_
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(ood_data[sentence_id])
random.seed(0)
annotations=[]
for i in range(len(clustered_sentences)):
    common = set(all_annotations) & set(clustered_sentences[i])
    #if len(common) <2:
    annotations.extend(random.sample(clustered_sentences[i],2))
all_annotations.extend(annotations)
labels_for_annotation=[]
for data in annotations:
    idx= valid_data.index(data)
    labels_for_annotation.append(valid_labels[idx])
    annotation_with_labels[data]=valid_labels[idx]
all_labels.extend(labels_for_annotation)
new_classes = len(set(labels_for_annotation)-set(train_labels))
print(f'No of new classes discovered:{new_classes}')


# In[ ]:


cluster_no={}
annotation_with_labels={}
from sklearn.cluster import KMeans
num_clusters = 64
random.seed(1)
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters,random_state=0)
# Fit the embedding with kmeans clustering.
clustering_model.fit(sen_embeddings)
cluster_assignment = clustering_model.labels_
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(ood_data[sentence_id])
random.seed(0)
annotations=[]
for i in range(len(clustered_sentences)):
    common = set(all_annotations) & set(clustered_sentences[i])
    #if len(common) <2:
    annotations.extend(random.sample(clustered_sentences[i],2))
all_annotations.extend(annotations)
labels_for_annotation=[]
for data in annotations:
    idx= valid_data.index(data)
    labels_for_annotation.append(valid_labels[idx])
    annotation_with_labels[data]=valid_labels[idx]
all_labels.extend(labels_for_annotation)
new_classes = len(set(labels_for_annotation)-set(train_labels))
print(f'No of new classes discovered:{new_classes}')


# # CQBA

# In[ ]:


from collections import defaultdict
cluster_data={}
cluster_data_annotation=defaultdict(list)
cluster_data_annotation_labels = defaultdict(list)
all_new_data=set()
for i in range(len(clustered_sentences)):
    data_pts = clustered_sentences[i]
    intersection = set(data_pts) & set(all_annotations)
    if len(intersection)==0:
        cluster_data[i].append([])
    else:
        labels_cluster=[]
        for ann in intersection:
            idx = all_annotations.index(ann)
            if all_labels[idx] not in train_labels:
                labels_cluster.append(all_labels[idx])
                cluster_data_annotation[i].append(ann)
                cluster_data_annotation_labels[i].append(all_labels[idx])
        cluster_data[i]=set(labels_cluster)
        for label in labels_cluster:
            all_new_data.add(label)


# In[ ]:


transform_cluster=defaultdict(list)
for cluster in cluster_data:
    values = cluster_data[cluster]
    for v in values:
        transform_cluster[v].append(cluster)


# In[ ]:


good_cluster=[]
bad_cluster=[]
for key in set(cluster_data_annotation_labels):
    if len(set(cluster_data[key]))==1:
        good_cluster.append(key)
    else:
        bad_cluster.append(key)


# In[ ]:


data_annotation_good_cluster = []
label_annotation_good_cluster=[]
annotation_embs=[]
data_annotation_bad_cluster=[]
label_annotation_bad_cluster=[]
bad_annotation_embs=[]
for cluster in good_cluster:
    for ann in cluster_data_annotation[cluster]:
        data_annotation_good_cluster.append(ann)
        annotation_embs.append(sen_embeddings[ood_data.index(ann)])
        idx = valid_data.index(ann)
        label_annotation_good_cluster.append(valid_labels[idx])
        #print(cluster)
for cluster in bad_cluster:
    for ann in cluster_data_annotation[cluster]:
        data_annotation_bad_cluster.append(ann)
        bad_annotation_embs.append(sen_embeddings[ood_data.index(ann)])
        idx = valid_data.index(ann)
        label_annotation_bad_cluster.append(valid_labels[idx])


# In[ ]:


not_good_clusters=[]
new_silver_data = []
new_silver_class = []
count=0
random.seed(1)
cluster_wise_pts_remaining=defaultdict(list)
for cluster in good_cluster:
    already_annotated = cluster_data_annotation[cluster]
    if len(already_annotated)<2:
        data = random.sample(clustered_sentences[cluster], 2- len(already_annotated))
        count+=len(data)
        data.extend(already_annotated)
    else:
        data = already_annotated
        #print(data)
    lbl_set=set()
    for d in data:
        lbls = new_lab_list[ood_data.index(d)]
        lbl_set.add(lbls)
    if len(lbl_set)==1:
        new_silver_data.extend(data)
        lbls_to_add = [list(lbl_set)[0]]*len(data)
        new_silver_class.extend(lbls_to_add)
        remaining_data = list(set(clustered_sentences[cluster])-set(data))
        cluster_wise_pts_remaining[cluster]=remaining_data
        f =open(f'output_cluster_wise/cluster_{cluster}_rem_pts.txt','w')
        for pts in remaining_data:
            lbl = new_lab_list[ood_data.index(pts)]
            f.write(pts.strip('\n')+'\t'+lbl)
        f.close()
    else:
        not_good_clusters.append(cluster)


# In[ ]:


new_bad_clusters = bad_cluster+ not_good_clusters
data_new_bad=[]
random.seed(1)
lbl_new_bad =[]
new_added=0
for cluster in set(new_bad_clusters):
    data_bad_cluster = cluster_data_annotation[cluster]
    lbl_bad_cluster = cluster_data_annotation_labels[cluster]
    data = cluster_data_annotation[cluster]
    #print(data_bad_cluster)
    data_new_bad.extend(data_bad_cluster)
    lbl_new_bad.extend(lbl_bad_cluster)
    print(len(data_bad_cluster))
    print(len(lbl_bad_cluster))
    if len(data_bad_cluster)<5:
        new_anns = random.sample(clustered_sentences[cluster],5-len(data_bad_cluster))
        data.extend(new_anns)
        new_added+=len(new_anns)
        for anns in new_anns:
            idx = ood_data.index(anns)
            ann_label = new_lab_list[idx]
            lbl_new_bad.append(ann_label)
            data_new_bad.append(anns)
    remaining_data = list(set(clustered_sentences[cluster])-set(data))
    cluster_wise_pts_remaining[cluster]=remaining_data
    f =open(f'output_cluster_wise/cluster_{cluster}_rem_pts.txt','w')
    for pts in remaining_data:
        lbl = new_lab_list[ood_data.index(pts)]
        f.write(pts.strip('\n')+'\t'+lbl)
    f.close()


# In[ ]:


new_data = new_silver_data+data_new_bad
new_labels = new_silver_class+lbl_new_bad


# In[ ]:


f = open('data_generated/new_data.txt','w')
for data in new_data:
    f.write(data)
f.close()
f = open('data_generated/new_class.txt','w')
for data in new_labels:
    f.write(data)
f.close()


# In[ ]:


all_annotated_clusters=defaultdict(list)
all_annotated_labels=defaultdict(list)
for c in cluster_data_annotation:
    annotated=set(clustered_sentences[c])-set(cluster_wise_pts_remaining[c])
    total =list(annotated)
    all_annotated_clusters[c]=total
    for t in total:
        lbl=new_lab_list[ood_data.index(t)]
        all_annotated_labels[c].append(lbl)


# In[ ]:


new_labelled_data=[]
new_labelled_class=[]
for data in all_annotated_clusters:
    text = all_annotated_clusters[data]
    lbl = all_annotated_labels[data]
    new_labelled_data.extend(text)
    new_labelled_class.extend(lbl)


# # PPAS

# In[ ]:


import os
random.seed(0)
count=0
total=0
path = "output_confidences/preds/"
for file in os.listdir(path):
    if file.startswith('gold'):
        f = open(os.path.join(path,file))
        lines = f.readlines()
        cluster_no=int(file.split('_')[3])
        high_conf=[]
        low_conf=[]
        for i in range(len(lines)):
            conf_score = float(lines[i])
            if conf_score>0.5:
                high_conf.append(cluster_wise_pts_remaining[cluster_no][i])
            else:
                low_conf.append(cluster_wise_pts_remaining[cluster_no][i])
        if len(low_conf)<2:
            sents=low_conf
        elif len(low_conf)>=2 and len(low_conf)<3:
            sents = random.sample(low_conf, 2)
        elif len(low_conf)>=3 and len(low_conf)<4:
            sents = random.sample(low_conf, 3)
        else:
            sents = random.sample(low_conf, 4)
        total+=len(sents)
        for sent in sents: #gold strategy
            lbls = new_lab_list[ood_data.index(sent)]
            all_annotated_clusters[cluster_no].append(sent)
            all_annotated_labels[cluster_no].append(lbls)
            new_labelled_data.append(sent)
            new_labelled_class.append(lbls)
        for pts in high_conf:
            utt_emb = sen_embeddings[ood_data.index(pts)]
            text = all_annotated_clusters[cluster_no]
            cos_sim=-10
            cos_scores=[]
            for t in text:
                item_emb = sen_embeddings[ood_data.index(t)]
                cosine_scores = util.pytorch_cos_sim(utt_emb, item_emb)
                cos_scores.append(cosine_scores)
            max_value = max(cos_scores)
            if max_value>0.8:
                idx = cos_scores.index(max_value)
                new_labelled_data.append(pts)
                new_labelled_class.append(all_annotated_labels[cluster_no][idx])


# In[ ]:


f = open('outputs_PPAS/new_data.txt','w')
for data in new_labelled_data:
    f.write(data)
f.close()
f = open('outputs_PPAS/new_class.txt','w')
for data in new_labelled_class:
    f.write(data)
f.close()

