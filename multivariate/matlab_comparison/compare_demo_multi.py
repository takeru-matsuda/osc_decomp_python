import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from examples.demo_multi import demo_multi
from function_test.calculate_mean_square_error import calculate_mean_square_error


# MATLAB の demo_multi の入出力値を読み込みます。
filename = parent_dir + '/matlab_test_outputs/demo_multi.mat'
data = scipy.io.loadmat(filename)
osc_AIC_m = np.array(data['osc_AIC'])
osc_mean_m = np.array(data['osc_mean'])
osc_cov_m = np.array(data['osc_cov'])
osc_phase_m = np.array(data['osc_phase'])
minAIC_m = np.array(data['minAIC']).item()
K_m = np.array(data['K']).item()
osc_a_m = np.array(data['osc_a'])
osc_f_m = np.array(data['osc_f'])
osc_sigma2_m = np.array(data['osc_sigma2'])
osc_c_m = np.array(data['osc_c'])
osc_tau2_m = np.array(data['osc_tau2']).item()
hess_m = np.array(data['hess'])
grad_m = np.array(data['grad'])
mll_m = np.array(data['mll']).item()
cov_est_m = np.array(data['cov_est'])

# osc_mean,osc_cov, osc_phase の範囲は、AIC 最小の K に制限します。
osc_mean_K_m = osc_mean_m[0:2*K_m, :, K_m-1]
osc_cov_K_m = osc_cov_m[0:2*K_m, 0:2*K_m, :, K_m-1]
osc_phase_K_m = osc_phase_m[0:K_m, :, K_m-1]

# Python の demo_multi を実行します。
[osc_AIC, osc_mean, osc_cov, osc_phase,
 minAIC, K, osc_a, osc_f, osc_sigma2, osc_c, osc_tau2,
 hess, grad, mll, cov_est] = demo_multi()

# osc_mean,osc_cov, osc_phase の範囲は、AIC 最小の K に制限します。
osc_mean_K = osc_mean[0:2*K, :, K-1]
osc_cov_K = osc_cov[0:2*K, 0:2*K, :, K-1]
osc_phase_K = osc_phase[0:K, :, K-1]

# 比較する変数と名称のリストを作成します。
var_list = [
    osc_AIC, osc_a, osc_f, osc_sigma2, osc_c, osc_tau2,
    osc_mean_K, osc_cov_K, osc_phase_K,
    hess, grad, mll, cov_est]
var_list_m = [
    osc_AIC_m, osc_a_m, osc_f_m, osc_sigma2_m, osc_c_m, osc_tau2_m,
    osc_mean_K_m, osc_cov_K_m, osc_phase_K_m,    
    hess_m, grad_m, mll_m, cov_est_m]
name_list = [
    'osc_AIC', 'osc_a', 'osc_f', 'osc_sigma2', 'osc_c', 'osc_tau2',    
    'osc_mean (K)', 'osc_cov (K)', 'osc_phase (K)',      
    'hess', 'grad', 'mll', 'cov_est']

# 変数の値を表示します。
np.set_printoptions(formatter={'float': '{:.5e}'.format})
for var, var_m, name in zip(var_list, var_list_m, name_list):
    print('Python result of {}:'.format(name), var)
    print('MATLAB result of {}:'.format(name), var_m)

# AIC 最小の K が同じ値であれば、変数の誤差を計算します。
if K==K_m:
    for var, var_m, name in zip(var_list, var_list_m, name_list):
        err_relative, err_absolute = calculate_mean_square_error(var, var_m)
        print('Square mean error of {}:'.format(name))
        print('relative {:.3e}:'.format(err_relative),
            'absolute {:.3e}:'.format(err_absolute))



