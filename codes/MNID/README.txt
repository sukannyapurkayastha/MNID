We provide the codes for using MNID in this folder:
1. The first step is the OOD Detection which we use the existing algorithm https://github.com/SLAD-ml/few-shot-ood 
2. We use the outcome from this algorithm stored in OOD_Data folder in MNID.py
3. Then we perform CQBA in MNID.py
4. Run PolyAI-Conf.py to find the confidence scores
5. Run MNID.py again to find the confidence based points
6. Run PolyAI.py finally to get the accuracies 