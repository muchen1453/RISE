import xgboost as xgb
import datetime
import ast
import shap
import os
import numpy as np
import shutil

def get_last_line(file_path):
    """ Reads the last line of a file safely. """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return lines[-1].strip() if lines else None
    except FileNotFoundError:
        print(f" Warning: Log file {file_path} not found.")
        return None
    except Exception as e:
        print(f" Error reading {file_path}: {e}")
        return None

train_path = f"../train"
shutil.rmtree(train_path, ignore_errors=True)
os.makedirs(train_path, exist_ok=True)

# Load test dataset efficiently
test_path = '../dataset/SOAP_testing.txt'
test = np.genfromtxt(test_path, usecols=[0], max_rows=2000, dtype=np.float32)

best_rmse = float("inf")
best_model = None

o_path = 'output.txt'
with open(o_path,'w') as o_file:
    for temp in range(7):
        o_file.write(f" Training Model {temp} - {datetime.datetime.now()}\n")

        para_path = f'temp{temp}/5fold-XGB-full.log'
        lastline = get_last_line(para_path)

        if lastline is None or 'Best XGBOOST parameters:' not in lastline:
            print(f" Skipping temp {temp} due to missing or malformed parameters.")
            continue

        parameter = lastline.split('Best XGBOOST parameters: ')[1]
        try:
            param = ast.literal_eval(parameter)
        except (SyntaxError, ValueError):
            print(f" Error parsing parameters for temp {temp}. Skipping.")
            continue

        # Ensure proper XGBoost settings
        param['max_depth'] = int(round(param.get('max_depth', 6)))  # Default to 6 if missing
        param['gpu_id'] = 0
        param['tree_method'] = 'gpu_hist'

        # Load training & validation data
        dtrain = xgb.DMatrix('../dataset/SOAP_training.txt?format=libsvm')
        dvalidate = xgb.DMatrix('../dataset/SOAP_validation.txt?format=libsvm')

        # Train the model
        evallist = [(dvalidate, 'eval'), (dtrain, 'train')]
        num_round = 100000
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)

        # Test the model
        dtest = xgb.DMatrix('../dataset/SOAP_testing.txt?format=libsvm')
        pred = bst.predict(dtest)

        # Compute RMSE
        rmse = np.sqrt(np.mean((pred - test) ** 2))
        o_file.write(f" RMSE of temp {temp}: {rmse:.6f}\n")

        # Save the best model only
        model_path = '../train/xgb.model'
        if rmse < best_rmse:
            if best_model and os.path.exists(best_model):
                os.remove(best_model)  # Delete only if it exists
            best_rmse = rmse
            best_model = model_path
            bst.save_model(best_model)  # Save only the best model
            error = np.array(pred - test)
            error_file = '../train/error.txt'
            np.savetxt(error_file,error)
            bst_bk = bst
            dtrain_bk = dtrain
            print(f"New best model saved at temp {temp} (RMSE: {best_rmse:.6f})\n")

    o_file.write(f" \nBest model at temp {temp} with RMSE: {best_rmse}")

###################### SHAP Analysis ######################
explainer = shap.TreeExplainer(bst_bk)
explanation = explainer(dtrain_bk)

if explanation.values is None:
    print(" Error: SHAP values could not be computed.")
    exit(1)

shap_values = explanation.values
shap_value = shap_values.mean(axis=0)

# Ensure SHAP file removal before writing
shap_file = '../train/shap_value.txt'
if os.path.exists(shap_file):
    os.remove(shap_file)

# Ensure ULE file exists
ule_file = '../LAEs/Atomic_configuration.txt'
if not os.path.exists(ule_file):
    print(f" Error: {ule_file} does not exist.")
    exit(1)

with open(ule_file, 'r') as ule, open(shap_file, 'w') as file:
    for i in range(len(shap_value)):
        line = ule.readline().strip()
        file.write(f"{line} {shap_value[i]:.6f}\n")

print(f" SHAP values saved to {shap_file}")
