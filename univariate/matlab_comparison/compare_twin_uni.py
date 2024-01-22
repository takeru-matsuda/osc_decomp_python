import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from examples.twin_uni import twin_uni
from function_test.calculate_mean_square_error import calculate_mean_square_error

# テスト 用の twin_uni の乱数の種です。
# 5489 または 1337 に対応しています。
seed = 5489
#seed = 1337

# MATLAB の twin_uni の入出力値を読み込みます。
filename = parent_dir + '/matlab_test_outputs/twin_uni{}.mat'.format(seed)
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

# Python の twin_uni を実行します。
[osc_AIC, osc_mean, osc_cov, osc_phase,
 minAIC, K, osc_a, osc_f, osc_sigma2, osc_tau2] = twin_uni(
    if_test=True, seed=seed)

# 比較する変数と名称のリストを作成します。
var_list = [
    osc_AIC, osc_a, osc_f, osc_sigma2, osc_tau2,
    osc_mean, osc_cov, osc_phase,
    ]
var_list_m = [
    osc_AIC_m, osc_a_m, osc_f_m, osc_sigma2_m, osc_tau2_m,
    osc_mean_m, osc_cov_m, osc_phase_m,
    ]
name_list = [
    'osc_AIC', 'osc_a', 'osc_f', 'osc_sigma2', 'osc_tau2',
    'osc_mean', 'osc_cov', 'osc_phase'
    ]

# 変数の誤差を計算します。
for var, var_m, name in zip(var_list, var_list_m, name_list):
    err_relative, err_absolute = calculate_mean_square_error(var, var_m)
    print('Square mean error of {}:'.format(name))
    print('relative {:.3e}:'.format(err_relative),
          'absolute {:.3e}:'.format(err_absolute))
