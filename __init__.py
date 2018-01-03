
""" must run in python3x"""
import  numpy as np
import tensorflow as tf
import os
import shutil
__author__ = 'Yu He'
__version__ = 'v30'

EVALUTION_THRESHOLD_FOR_MULTI_LABEL = 0.5


detail_filename = os.path.join('./data', 'best_eval_for_predicted_value_dictribution')
total_predicted_value_dictribution = np.loadtxt(detail_filename,dtype=float)
detail_filename = os.path.join('./data', 'best_eval_for_true_value')
total_true_value = np.loadtxt(detail_filename,dtype=int)

total_predicted_value = ((total_predicted_value_dictribution) >= EVALUTION_THRESHOLD_FOR_MULTI_LABEL).astype(int)



# label34 = np.ones([total_true_value.shape[0],17],dtype=int)
# total_true_value = np.concatenate((total_true_value,label34),axis=1)
# total_predicted_value = np.concatenate((total_predicted_value,label34),axis=1)
#


filename_eval_log = os.path.join('./data', 'log_eval')
file_eval_log = open(filename_eval_log, 'w')
np.set_printoptions(threshold=np.nan)
print('\nevaluation:', file=file_eval_log)
print('\nevaluation:')

total_predicted_value = total_predicted_value.astype(bool)
total_true_value = total_true_value.astype(bool)

print('  example based evaluations:', file=file_eval_log)
print('  example based evaluations:')

equal = total_true_value == total_predicted_value
match = np.sum(equal, axis=1) == np.size(equal, axis=1)
exact_match_ratio = np.sum(match) / np.size(match)
print('      exact_match_ratio = %.4f' % exact_match_ratio, file=file_eval_log)
print('      exact_match_ratio = %.4f' % exact_match_ratio)

true_and_predict = np.sum(total_true_value & total_predicted_value, axis=1)
true_or_predict = np.sum(total_true_value | total_predicted_value, axis=1)
accuracy = np.mean(true_and_predict / true_or_predict)
print('      accuracy = %.4f' % accuracy, file=file_eval_log)
print('      accuracy = %.4f' % accuracy)

precison = np.mean(true_and_predict / (np.sum(total_predicted_value, axis=1) + 1e-9))
print('      precison = %.4f' % precison, file=file_eval_log)
print('      precison = %.4f' % precison)

recall = np.mean(true_and_predict / np.sum(total_true_value, axis=1))
print('      recall = %.4f' % recall, file=file_eval_log)
print('      recall = %.4f' % recall)

F1_Measure = np.mean((true_and_predict * 2) / (np.sum(total_true_value, axis=1)
                                               + np.sum(total_predicted_value, axis=1)))
print('      F1_Measure = %.4f' % F1_Measure, file=file_eval_log)
print('      F1_Measure = %.4f' % F1_Measure)

HammingLoss = np.mean(total_true_value ^ total_predicted_value)
print('      HammingLoss = %.4f' % HammingLoss, file=file_eval_log)
print('      HammingLoss = %.4f' % HammingLoss)


print('  label based evaluations:', file=file_eval_log)
print('  label based evaluations:')

TP = np.sum(total_true_value & total_predicted_value,axis=0,dtype=np.int32)
FP = np.sum((~total_true_value) & total_predicted_value,axis=0,dtype=np.int32)
FN = np.sum(total_true_value & (~total_predicted_value),axis=0,dtype=np.int32)

TP_re = np.reshape(TP,[TP.shape[0],1])
FP_re = np.reshape(FP,[FP.shape[0],1])
FN_re = np.reshape(FN,[FN.shape[0],1])
re =  np.concatenate((TP_re,FP_re,FN_re),axis=1)
print('TP FP FN:')
print('TP FP FN:', file=file_eval_log)
print(re,file=file_eval_log)
print(re)


# TP = np.concatenate((TP[0:6],TP[7:28],TP[29:31],TP[32:36],TP[37:52],TP[53:]))
# FP = np.concatenate((FP[0:6],FP[7:28],FP[29:31],FP[32:36],FP[37:52],FP[53:]))
# FN = np.concatenate((FN[0:6],FN[7:28],FN[29:31],FN[32:36],FN[37:52],FN[53:]))

# for i in [6,28,31,36,52]:
#     TP[i] = TP[i-1]
#     FP[i] = FP[i - 1]
#     FN[i] = FN[i - 1]
#
# TP = np.concatenate((TP[0:49],TP[51:66],TP[67:69],TP[70:80],TP[81:]))
# FP = np.concatenate((FP[0:49],FP[51:66],FP[67:69],FP[70:80],FP[81:]))
# FN = np.concatenate((FN[0:49],FN[51:66],FN[67:69],FN[70:80],FN[81:]))


_P = np.sum(TP) / (np.sum(TP) + np.sum(FP)  + 1e-9 )
_R = np.sum(TP) / (np.sum(TP) + np.sum(FN)  + 1e-9 )
Micro_F1 = (2 * _P *_R) / (_P + _R)
print('      P = %.4f' % _P, file=file_eval_log)
print('      P = %.4f' % _P)
print('      R = %.4f' % _R, file=file_eval_log)
print('      R = %.4f' % _R)
print('      Micro-F1 = %.4f' % Micro_F1, file=file_eval_log)
print('      Micro-F1 = %.4f' % Micro_F1)

_P_t = TP / (TP + FP + 1e-9)
_R_t = TP / (TP + FN + 1e-9)
Macro_F1 = np.mean((2 * _P_t * _R_t) / (_P_t + _R_t + 1e-9))


_P_t_re = np.reshape(_P_t,[_P_t.shape[0],1])
_R_t_re = np.reshape(_R_t,[_R_t.shape[0],1])
re =  np.concatenate((_P_t_re,_R_t_re),axis=1)
print('_P_t _R_t:')
print('_P_t:', file=file_eval_log)
print(re,file=file_eval_log)
print(re)

print('      Macro-F1 = %.4f' % Macro_F1, file=file_eval_log)
print('      Macro-F1 = %.4f' % Macro_F1)
