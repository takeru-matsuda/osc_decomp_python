import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from examples.opt_cmp_uni import opt_cmp_uni
from function_test.calculate_mean_square_error import calculate_mean_square_error

# MATLAB の opt_cmp_uni の入出力値を読み込みます。
filename = parent_dir + '/matlab_test_outputs/opt_cmp_uni.mat'
data = scipy.io.loadmat(filename)
osc_param1_m = np.array(data['osc_param1'])
osc_AIC1_m = np.array(data['osc_AIC1'])
osc_param2_m = np.array(data['osc_param2'])
osc_AIC2_m = np.array(data['osc_AIC2'])
osc_param3_m = np.array(data['osc_param3'])
osc_AIC3_m = np.array(data['osc_AIC3'])

# Python の opt_cmp_uni を実行します。
[osc_param1, osc_AIC1,
 osc_param2, osc_AIC2, 
 osc_param3, osc_AIC3] = opt_cmp_uni()

# 比較する変数と名称のリストを作成します。
var_list = [
    osc_AIC1, osc_param1,
    osc_AIC2, osc_param2, 
    osc_AIC3, osc_param3, ]

var_list_m = [
    osc_AIC1_m, osc_param1_m,
    osc_AIC2_m, osc_param2_m, 
    osc_AIC3_m, osc_param3_m, ]

name_list = [
    'osc_AIC1', 'osc_param1',
    'osc_AIC2', 'osc_param2', 
    'osc_AIC3', 'osc_param3', ]

# 変数の誤差を計算します。
for var, var_m, name in zip(var_list, var_list_m, name_list):
    err_relative, err_absolute = calculate_mean_square_error(var, var_m)
    print('Square mean error of {}:'.format(name))
    print('relative {:.3e}:'.format(err_relative),
          'absolute {:.3e}:'.format(err_absolute))

# osc_param1,2,3 の間の誤差を計算します。
err13 = calculate_mean_square_error(osc_param1, osc_param3)[0]
err12 = calculate_mean_square_error(osc_param1, osc_param2)[0]
err23 = calculate_mean_square_error(osc_param2, osc_param3)[0]
print('err13 = ', err13)
print('err12 = ', err12)
print('err23 = ', err23)
