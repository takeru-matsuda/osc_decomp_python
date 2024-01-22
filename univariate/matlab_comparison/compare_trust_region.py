import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from pyoscdecomp.osc_decomp import osc_decomp
from function_test.calculate_mean_square_error import calculate_mean_square_error

data = scipy.io.loadmat(parent_dir + '/matlab_test_outputs/osc_decomp_uniinputs.mat')
Y = np.array(data['Y'])
fs = np.array(data['fs']).item()

MAX_OSC = 6
MAX_AR = 21
algorithm = 'trust-region'
grad = True

# MATLAB の信頼領域法の結果を読み込みます。
matlab_file = parent_dir + '/matlab_test_outputs/osc_decomp_2.mat'
data = scipy.io.loadmat(matlab_file)
osc_param_tr = np.array(data['osc_param'])
osc_AIC_tr = np.array(data['osc_AIC'])
osc_mean_tr = np.array(data['osc_mean'])
osc_cov_tr = np.array(data['osc_cov'])
osc_phase_tr = np.array(data['osc_phase'])

# MATLAB の準 Newton 法の結果を読み込みます。
matlab_file = parent_dir + '/matlab_test_outputs/osc_decomp_1.mat'
data = scipy.io.loadmat(matlab_file)
osc_param_qN = np.array(data['osc_param'])
osc_AIC_qN = np.array(data['osc_AIC'])
osc_mean_qN = np.array(data['osc_mean'])
osc_cov_qN = np.array(data['osc_cov'])
osc_phase_qN = np.array(data['osc_phase'])

# Python の osc_decomp_uni を実行します。
[osc_param, 
 osc_AIC, 
 osc_mean, 
 osc_cov,
 osc_phase] = osc_decomp(Y, 
                         fs, 
                         MAX_OSC=MAX_OSC, 
                         MAX_AR=MAX_AR, 
                         algorithm=algorithm, 
                         grad=grad)
# 比較する変数のリストです。
var_list = [
    osc_param, osc_AIC, osc_mean, osc_cov, osc_phase]
var_list_tr = [
    osc_param_tr, osc_AIC_tr, osc_mean_tr, osc_cov_tr, osc_phase_tr]
var_list_qN = [
    osc_param_qN, osc_AIC_qN, osc_mean_qN, osc_cov_qN, osc_phase_qN]
name_list = [
    'osc_param', 'osc_AIC', 'osc_mean', 'osc_cov', 'osc_phase']

relative_errlist = []
absolute_errlist = []

# MATLAB の信頼領域法の結果との誤差を表示します。
print('==================== trust-region ====================')
for var, var_m, name in zip(var_list, var_list_tr, name_list):
    err_relative, err_absolute = calculate_mean_square_error(var, var_m)
    print('Square mean error of {}:'.format(name))
    print('relative {:.3e}:'.format(err_relative),
        'absolute {:.3e}:'.format(err_absolute))
    relative_errlist.append(err_relative)
    absolute_errlist.append(err_absolute)

relative_errlist = np.array(relative_errlist)
absolute_errlist = np.array(absolute_errlist)

# MATLAB の準 Newton 法の結果との誤差を表示します。
print('==================== quasi-newton ====================')
relative_errlist = []
absolute_errlist = []
for var, var2, name in zip(var_list, var_list_qN, name_list):
    err_relative, err_absolute = calculate_mean_square_error(var, var2)
    print('Square mean error of {}:'.format(name))
    print('relative {:.3e}:'.format(err_relative),
        'absolute {:.3e}:'.format(err_absolute))
    relative_errlist.append(err_relative)
    absolute_errlist.append(err_absolute)

relative_errlist = np.array(relative_errlist)
absolute_errlist = np.array(absolute_errlist)
