import sys
import math
from workspace import workspace
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

def search_best_threshold(params, valid_output_info_list):
    dataset_best_thresholds = []
    dataset_best_values = []

    for valid_output_info in valid_output_info_list:
        bestT = 0
        bestV = 1
        best_frr = 0
        best_far = 1

        offsize = len([conf for pred, gt, conf in valid_output_info
                      if gt == params['offtopic_label']])
        insize = len([conf for pred, gt, conf in valid_output_info
                     if gt != params['offtopic_label']])

        print('\noffsize, insize', offsize, insize)
        sorted_valid_output_info = sorted(valid_output_info, key=lambda x: x[2])

        accepted_oo = offsize
        rejected_in = 0.0
        threshold = 0.0
        ind = 0
        for pred, gt, conf in sorted_valid_output_info[:-1]:
            threshold = (sorted_valid_output_info[ind][2] + 
                         sorted_valid_output_info[ind+1][2])/2.0
            if gt != params['offtopic_label']:
                rejected_in += 1.0
            else:
                accepted_oo -= 1.0

            frr = rejected_in / insize
#             if offsize==0: offsize=1
            far = accepted_oo / offsize
            dist = math.fabs(frr - far)
            if dist < bestV:
                bestV = dist
                bestT = threshold
                best_frr = frr
                best_far = far
            ind += 1

        dataset_best_thresholds.append(bestT)
        dataset_best_values.append(bestV)
        print('bestT, bestV, bestFAR, bestFRR', 
              bestT, bestV, best_far, best_frr)

    return dataset_best_thresholds, dataset_best_values


def get_results(params, output_info, threshold):
#     print("*"*50)
#     print(params['offtopic_label'], threshold)
#     print()
#     for i in output_info: print(i)
#     print()
    total_gt_ontopic_utt = len([gt for pred, gt, conf in output_info
                               if gt != params['offtopic_label']])
    total_gt_offtopic_utt = len(output_info) - total_gt_ontopic_utt
#     print(len(output_info), total_gt_ontopic_utt, total_gt_offtopic_utt)
#     print("*"*50)
#     a = input()
    accepted_oo = 0.0
    rejected_in = 0.0
    correct_domain_label = 0.0
    correct_wo_thr = 0.0
    correct_w_thr = 0.0

    for pred, gt, conf in output_info:
        if conf < threshold:
            pred1 = params['offtopic_label']
        else:
            pred1 = pred

        if gt == params['offtopic_label'] and pred1 != gt:
            accepted_oo += 1
        elif gt != params['offtopic_label'] and pred1 == params['offtopic_label']:
            rejected_in += 1
        else:
            correct_domain_label += 1

        if gt != params['offtopic_label'] and pred == gt:
            correct_wo_thr += 1
        if gt != params['offtopic_label'] and pred1 == gt:
            correct_w_thr += 1

#     if total_gt_offtopic_utt==0: total_gt_offtopic_utt = 1      
    far = accepted_oo / total_gt_offtopic_utt
    frr = rejected_in / total_gt_ontopic_utt
    eer = 1 - correct_domain_label / len(output_info)
    ontopic_acc_ideal = correct_wo_thr / total_gt_ontopic_utt
    ontopic_acc = correct_w_thr / total_gt_ontopic_utt

    return eer, far, frr, ontopic_acc_ideal, ontopic_acc


def compute_values(params, experiment, result_data, epoch):

    t_macro_avg_eer = 0.0
    t_macro_avg_far = 0.0
    t_macro_avg_frr = 0.0

    t_macro_avg_acc_ideal = 0.0
    t_macro_avg_acc = 0.0

    test_output_info_list = []

    for workspace_idx in range(len(result_data)):
        curr_dev_workspace = result_data[workspace_idx]
        _, _, test_output_info = \
            experiment.run_testing_epoch(epoch=epoch,
                                         test_workspace=curr_dev_workspace)
        test_output_info_list.append(test_output_info)

    thesholds, _ = search_best_threshold(params, test_output_info_list)

    for workspace_idx in range(len(result_data)):
        curr_dev_workspace = result_data[workspace_idx]
        print('\n', curr_dev_workspace.target_sets_files[2])

        test_output_info = test_output_info_list[workspace_idx]
        ##################
        threshold = thesholds[workspace_idx]
        y_true, y_pred = [], []
        for pred, gt, conf in test_output_info:
            y_true.append(gt)
            y_pred.append(params['offtopic_label'] if conf<threshold else pred)
        
        uniq_lbls = sorted(list(set(y_true+y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=uniq_lbls)
        print('\tConfusion Matrix')
        for i in cm: print('\t',i)
        
        
        cls_acc = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()
        digits=4
        print('\tClasswise Accuracies')
        for l,a in zip(uniq_lbls, list(cls_acc)): 
            print(l.rjust(50)+'\t'+str(round(a,digits)))
        print(f'\tAccuracy = {accuracy_score(y_true, y_pred)}')
#         clf_report = classification_report(y_true, y_pred, digits=4, target_names=LABELS)
        clf_report = classification_report(y_true, y_pred, digits=4, labels=uniq_lbls)
        print("\t"+clf_report.replace("\n", "\n\t"))
        
        
        
        
        
        
        
        
        y_true = ['OOD' if i=='UNCONFIDENT_INTENT_FROM_SLAD' else 'IND' for i in y_true]
        y_pred = ['OOD' if i=='UNCONFIDENT_INTENT_FROM_SLAD' else 'IND' for i in y_pred]
        print("*"*30, 'IND-OOD', "*"*30)
        uniq_lbls = ['IND', 'OOD']
        cm = confusion_matrix(y_true, y_pred, labels=uniq_lbls)
        print('\tConfusion Matrix')
        for i in cm: print('\t',i)
        
        
        cls_acc = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()
        digits=4
        print('\tClasswise Accuracies')
        for l,a in zip(uniq_lbls, list(cls_acc)): 
            print(l.rjust(50)+'\t'+str(round(a,digits)))
        print(f'\tAccuracy = {accuracy_score(y_true, y_pred)}')
#         clf_report = classification_report(y_true, y_pred, digits=4, target_names=LABELS)
        clf_report = classification_report(y_true, y_pred, digits=4, labels=uniq_lbls)
        print("\t"+clf_report.replace("\n", "\n\t"))
        
        
        ##################
        test_eer, test_far, test_frr, test_ontopic_acc_ideal, \
            test_ontopic_acc = get_results(params, test_output_info, 
                                           thesholds[workspace_idx])
        print('test(eer, far, frr, ontopic_acc_ideal, ontopic_acc) %.3f, %.3f, %.3f, %.3f, %.3f' %
              (test_eer, test_far, test_frr,
               test_ontopic_acc_ideal,
               test_ontopic_acc))

        t_macro_avg_eer += test_eer
        t_macro_avg_far += test_far
        t_macro_avg_frr += test_frr
        t_macro_avg_acc_ideal += test_ontopic_acc_ideal
        t_macro_avg_acc += test_ontopic_acc

    t_macro_avg_eer /= len(result_data)
    t_macro_avg_far /= len(result_data)
    t_macro_avg_frr /= len(result_data)
    t_macro_avg_acc_ideal /= len(result_data)
    t_macro_avg_acc /= len(result_data)
    return t_macro_avg_eer, t_macro_avg_far, t_macro_avg_frr, \
        t_macro_avg_acc_ideal, t_macro_avg_acc, test_output_info_list


def get_data(params, file_list, role, period):
    workspaces = []
    with open(file_list) as fi:
        i = 1
        for enum, wid in enumerate(fi):
            if (f'Period{period}' not in wid) and ('train' not in role): continue
            wid = wid.strip().split('\t')[0]
            workspaces.append(workspace(wid, params, role))
            print(f'get_data:{i}'.ljust(20), wid.ljust(60), role.split('_')[0])
            sys.stdout.flush()
            i += 1
    return workspaces
