import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from examples.demo_uni import demo_uni
from function_test.calculate_mean_square_error import calculate_mean_square_error


# MATLAB の demo_uni の入出力値を読み込みます。
filename = parent_dir + '/matlab_test_outputs/demo_uni.mat'
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
osc_tau2_m = np.array(data['osc_tau2']).item()
hess_m = np.array(data['hess'])
grad_m = np.array(data['grad'])
mll_m = np.array(data['mll']).item()
cov_est_m = np.array(data['cov_est'])

# Python の demo_uni を実行します。
[osc_AIC, osc_mean, osc_cov, osc_phase,
 minAIC, K, osc_a, osc_f, osc_sigma2, osc_tau2,
 hess, grad, mll, cov_est] = demo_uni()

# 比較する変数と名称のリストを作成します。
var_list = [
    osc_AIC, osc_a, osc_f, osc_sigma2, osc_tau2,
    osc_mean, osc_cov, osc_phase,
    hess, grad, mll, cov_est]
var_list_m = [
    osc_AIC_m, osc_a_m, osc_f_m, osc_sigma2_m, osc_tau2_m,
    osc_mean_m, osc_cov_m, osc_phase_m,    
    hess_m, grad_m, mll_m, cov_est_m]
name_list = [
    'osc_AIC', 'osc_a', 'osc_f', 'osc_sigma2', 'osc_tau2',    
    'osc_mean', 'osc_cov', 'osc_phase',      
    'hess', 'grad', 'mll', 'cov_est']

# 変数の誤差を計算します。
for var, var_m, name in zip(var_list, var_list_m, name_list):
    err_relative, err_absolute = calculate_mean_square_error(var, var_m)
    print('Square mean error of {}:'.format(name))
    print('relative {:.3e}:'.format(err_relative),
          'absolute {:.3e}:'.format(err_absolute))

