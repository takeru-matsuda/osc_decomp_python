import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from examples.opt_cmp_multi import opt_cmp_multi
from function_test.calculate_mean_square_error import calculate_mean_square_error

# MATLAB の opt_cmp_uni の入出力値を読み込みます。
filename = parent_dir + '/matlab_test_outputs/opt_cmp_multi.mat'
data = scipy.io.loadmat(filename)

# 比較用の MATLAB 出力値です。
# quasi-Newton + False
osc_param1_m = np.array(data['osc_param1'])
osc_AIC1_m = np.array(data['osc_AIC1'])
# quasi-Newton + True
osc_param2_m = np.array(data['osc_param2'])
osc_AIC2_m = np.array(data['osc_AIC2'])
# trust-region + True
osc_param3_m = np.array(data['osc_param3'])
osc_AIC3_m = np.array(data['osc_AIC3'])
# trust-region-ncg の結果です。quasi-Newton + False と比較します。
osc_param4_m = osc_param1_m
osc_AIC4_m = osc_AIC1_m

# Python の opt_cmp_multi を実行します。
[osc_param1, osc_AIC1,
 osc_param2, osc_AIC2, 
 osc_param3, osc_AIC3,
 osc_param4, osc_AIC4] = opt_cmp_multi()

np.set_printoptions(formatter={'float': '{:.5e}'.format})
print('osc_AIC1 = ', osc_AIC1)
print('osc_AIC1_m = ', osc_AIC1_m)
print('osc_AIC2 = ', osc_AIC2)
print('osc_AIC2_m = ', osc_AIC2_m)
print('osc_AIC3 = ', osc_AIC3)
print('osc_AIC3_m = ', osc_AIC3_m)
print('osc_AIC4 = ', osc_AIC4)
print('osc_AIC4_m = ', osc_AIC4_m)


# K = 1, 2, ..., MAX_OSC までの各最適化結果を比較します。
MAX_OSC = 6
J = 2
for K in range(1, MAX_OSC+1):
    print('Oscilator number = ', K)
    # 比較する変数と名称のリストを作成します。
    var_list = [
        osc_AIC1[0, K-1], osc_param1[K-1:K, 0:(2*J+1)*K+1],
        osc_AIC2[0, K-1], osc_param2[K-1:K, 0:(2*J+1)*K+1], 
        osc_AIC3[0, K-1], osc_param3[K-1:K, 0:(2*J+1)*K+1],
        osc_AIC4[0, K-1], osc_param4[K-1:K, 0:(2*J+1)*K+1], ]

    # 比較対象の MATLAB データは準 Newton 法と
    # 数値微分勾配で得られた結果とします。
    var_list_m = [
        osc_AIC1_m[0, K-1], osc_param1_m[K-1:K, 0:(2*J+1)*K+1],
        osc_AIC1_m[0, K-1], osc_param1_m[K-1:K, 0:(2*J+1)*K+1], 
        osc_AIC1_m[0, K-1], osc_param1_m[K-1:K, 0:(2*J+1)*K+1], 
        osc_AIC1_m[0, K-1], osc_param1_m[K-1:K, 0:(2*J+1)*K+1], ]

    name_list = [
        'osc_AIC1', 'osc_param1',
        'osc_AIC2', 'osc_param2', 
        'osc_AIC3', 'osc_param3',
        'osc_AIC4', 'osc_param4', ]

    # 変数の誤差を計算します。
    for var, var_m, name in zip(var_list, var_list_m, name_list):
        err_relative, err_absolute = calculate_mean_square_error(var, var_m)
        print('Square mean error of {}:'.format(name))
        print('relative {:.3e}:'.format(err_relative),
            'absolute {:.3e}:'.format(err_absolute))

