import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from bayes_opt import BayesianOptimization
import warnings
from bayes_opt import UtilityFunction

dtrain = xgb.DMatrix('../../dataset/SOAP_training.txt?format=libsvm')
# Comment out any parameter you don't want to test
def XGB_CV(
          max_depth,
          eta,
          gamma,
          min_child_weight,
          subsample,
          colsample_bytree
         ):

    global RMSEbest
    global ITERbest

#
# Define all XGboost parameters
#

    paramt = {
              'gpu_id' : 0,
              'tree_method' : 'gpu_hist',
              'predictor' : 'gpu_predictor',

              'booster' : 'gbtree',
              'max_depth' : int(max_depth),
              'gamma' : gamma,
              'eta' : eta,
              'objective' : 'reg:squarederror',
              # 'nthread' : 4,
              # 'silent' : True,
              'eval_metric': 'rmse',
              'subsample' : max(min(subsample, 1), 0),
              'colsample_bytree' : max(min(colsample_bytree, 1), 0),
              'min_child_weight' : min_child_weight,
              # 'max_delta_step' : int(max_delta_step),
              # 'seed' : 1001
              }

    folds = 5

    print("\n Search parameters (%d-fold validation):\n %s" % (folds, paramt), file=log_file )
    log_file.flush()

    xgbc = xgb.cv(
                    paramt,
                    dtrain,
                    num_boost_round = 20000,
                    # stratified = True,
                    nfold = folds,
#                    verbose_eval = 10,
                    early_stopping_rounds = 100,
                    metrics = 'rmse',
                    show_stdv = True
               )

# This line would have been on top of this section
#    with capture() as result:

# After xgb.cv is done, this section puts its output into log file. Train and validation scores
# are also extracted in this section. Note the "diff" part in the printout below, which is the
# difference between the two scores. Large diff values may indicate that a particular set of
# parameters is overfitting, especially if you check the CV portion of it in the log file and find
# out that train scores were improving much faster than validation scores.

#    print('', file=log_file)
#    for line in result[1]:
#        print(line, file=log_file)
#    log_file.flush()

    # print(xgbc)
    val_score = xgbc['test-rmse-mean'].iloc[-1]
    train_score = xgbc['train-rmse-mean'].iloc[-1]
    print(' Stopped after %d iterations with train-rmse = %f val-rmse = %f ( diff = %f )' % ( len(xgbc), train_score, val_score, (val_score - train_score)))
    print('\n Stopped after %d iterations with train-rmse = %f val-rmse = %f ( diff = %f )\n' % ( len(xgbc), train_score, val_score, (val_score - train_score)), file=log_file)
    log_file.flush()
    if (val_score < RMSEbest ):
        RMSEbest = val_score
        ITERbest = len(xgbc)

    return -val_score


# Define the log file. If you repeat this run, new output will be added to it
log_file = open('5fold-XGB-full.log', 'a')
RMSEbest = float('inf')
ITERbest = 0

XGB_BO = BayesianOptimization(XGB_CV, {
                                     'max_depth': (10, 50),
                                     'eta': (0.001,0.05),
                                     'gamma': (0.0001, 10.0),
                                     'min_child_weight': (0, 800),
                                     'subsample': (0.1, 1.0),
                                     'colsample_bytree' :(0.1, 1.0)
                                    })


print('-'*130)
print('-'*130, file=log_file)
log_file.flush()

acquisition_function = UtilityFunction(kind="ei", xi=1e-3)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
       
    XGB_BO.maximize(init_points=10, n_iter=20, acquisition_function=acquisition_function)

print('-'*130)
print('Final Results')
print('Maximum XGBOOST value: %f' % XGB_BO.max['target'])
print('Best XGBOOST parameters: ', XGB_BO.max['params'])
print('-'*130, file=log_file)
print('Final Result:', file=log_file)
print('Maximum XGBOOST value: %f' % XGB_BO.max['target'], file=log_file)
print('Best XGBOOST parameters: ', XGB_BO.max['params'], file=log_file)
log_file.flush()
log_file.close()
