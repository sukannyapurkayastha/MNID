import os
import pickle


with open("cluster_wise_pts_remaining.pkl", "rb") as pkl_handle:
    cluster_wise_pts_remaining = pickle.load(pkl_handle)

with open("new_lab_list.pkl", "rb") as pkl_handle:
    new_lab_list = pickle.load(pkl_handle)

with open("ood_data.pkl", "rb") as pkl_handle:
    ood_data = pickle.load(pkl_handle)

with open("all_annotated_clusters.pkl", "rb") as pkl_handle:
    all_annotated_clusters = pickle.load(pkl_handle)

with open("all_annotated_labels.pkl", "rb") as pkl_handle:
    all_annotated_labels = pickle.load(pkl_handle)

with open("new_labelled_data.pkl", "rb") as pkl_handle:
    new_labelled_data = pickle.load(pkl_handle)

with open("new_labelled_class.pkl", "rb") as pkl_handle:
    new_labelled_class = pickle.load(pkl_handle)

with open("budget.pkl", "rb") as pkl_handle:
    budget = pickle.load(pkl_handle)




random.seed(0)
count=0
total=0
annotated=0
path = "output_confidences/preds/"
for file in os.listdir(path):
    if file.startswith('gold'):
        f = open(os.path.join(path,file))
        lines = f.readlines()
        cluster_no=int(file.split('_')[3])
        high_conf=[]
        low_conf=[]
        low_conf_scores = []
        for i in range(len(lines)):
            conf_score = float(lines[i])
            if conf_score>0.5:
                high_conf.append(cluster_wise_pts_remaining[cluster_no][i])
            else:
                low_conf.append(cluster_wise_pts_remaining[cluster_no][i])
                low_conf_scores.append(conf_score)
        low_conf_scores, low_conf = zip(*sorted(zip(low_conf_scores, low_conf_scores)))
        low_conf  = low_conf[:budget]
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