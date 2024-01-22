import pytest
import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
sys.path.append(current_dir)
from pyoscdecomp.osc_decomp import osc_decomp
from calculate_mean_square_error import calculate_mean_square_error

# 誤差の閾値です。
decimal = 5
tol = np.float64(1.5 * 10**(-decimal))

# osc_decomp のテスト ケースです。
# 比較する MATLAB ファイルは、'quasi-Newton' と grad = False の結果とします。
@pytest.mark.parametrize('dim, MAX_OSC, MAX_AR, algorithm, grad, matlab_file', [
#    (1, 6, 21, 'quasi-newton', True, 'matlab_test_outputs/osc_decomp_3.mat'),  # 単一時系列 (準 Newton、Kalman-filter)
#    (1, 6, 21, 'trust-region', True, 'matlab_test_outputs/osc_decomp_3.mat'),  # 単一時系列 (信頼領域)
#    (1, 6, 21, 'quasi-newton', False, 'matlab_test_outputs/osc_decomp_3.mat'),  # 単一時系列 (準 Newton、数値微分)
#    (1, 6, 21, 'trust-region-ncg', True, 'matlab_test_outputs/osc_decomp_3.mat'),  # 単一時系列 (信頼領域 Newton-CG)
    (2, 6, 21, 'quasi-newton', True, 'matlab_test_outputs/osc_decomp_var_3.mat'), # 複合時系列 (準 Newton、Kalman-filter)
    (2, 6, 21, 'trust-region', True, 'matlab_test_outputs/osc_decomp_var_3.mat'), # 複合時系列 (信頼領域)
    (2, 6, 21, 'quasi-newton', False, 'matlab_test_outputs/osc_decomp_var_3.mat'),  # 複合時系列 (準 Newton、数値微分)
    (2, 6, 21, 'trust-region-ncg', True, 'matlab_test_outputs/osc_decomp_var_3.mat'),  # 複合時系列 (信頼領域 Newton-CG)
    (2, None, None, None, None, 'matlab_test_outputs/osc_decomp_var_3.mat'),  # 複合時系列のデフォルト値です。
    (1, None, None, None, None, 'matlab_test_outputs/osc_decomp_uni_3.mat'),  # 単一時系列のデフォルト値です。
    (1, None, None, 'Nelder-Mead', None, 'matlab_test_outputs/osc_decomp_var_3.mat')  # 未対応の最適化です。
])
def test_osc_decomp(dim, MAX_OSC, MAX_AR, algorithm, grad, matlab_file):
    """
    osc_decomp のテストです。
    引数ごとにテストを行います。
    """    
    print('dimension = ', dim, 'algorithm = ', algorithm, 'grad = ', grad)
    if dim == 2:
        input_file = parent_dir + '/NorthSouthSunspotData.mat'
        matdata = scipy.io.loadmat(input_file)
        dat = matdata['dat']

        Y = np.log(dat + 1)
        J = Y.shape[0]
        T = Y.shape[1]
        Y = Y - np.mean(Y, axis=1).reshape(J, 1) @ np.ones((1, T))
        fs = 12
    elif dim == 1:
        input_file = parent_dir + '/CanadianLynxData.mat'
        matdata = scipy.io.loadmat(input_file)
        lynx = matdata['lynx']
        Y = np.log(lynx)
        Y = Y - np.mean(Y)
        J = Y.shape[0]
        T = Y.shape[1]
        fs = 1.0

    print('Y.shape = ', Y.shape)
    
    data = scipy.io.loadmat(parent_dir + '/' + matlab_file)
    osc_param_m = np.array(data['osc_param'])
    osc_AIC_m = np.array(data['osc_AIC'])
    osc_mean_m = np.array(data['osc_mean'])
    osc_cov_m = np.array(data['osc_cov'])
    osc_phase_m = np.array(data['osc_phase'])
    print('MAX_OSC = ', MAX_OSC)

    # Python の osc_decomp を実行します。
    # 未対応の最適化方法が入力された場合に ValueError の送出をアサートします。
    with pytest.raises(ValueError) as e:
        [
            osc_param, 
            osc_AIC, 
            osc_mean, 
            osc_cov,
            osc_phase] = osc_decomp(
                Y, fs, 
                MAX_OSC=MAX_OSC, 
                MAX_AR=MAX_AR, 
                algorithm=algorithm, 
                grad=grad)

        np.set_printoptions(formatter={'float': '{:.3e}'.format})
        print('osc_AIC = ', osc_AIC)
        print('osc_AIC_m = ', osc_AIC_m)
        K_py = np.nanargmin(osc_AIC) + 1
        K_mat = np.nanargmin(osc_AIC_m) + 1
        minAIC = np.nanmin(osc_AIC)
        minAIC_m = np.nanmin(osc_AIC_m)

        print('K, minAIC = ',K_py, minAIC)
        print('K_mat, minAIC_m = ',K_mat, minAIC_m)

        # AIC 最小の K が同一であることのアサーションです。
        assert (K_py == K_mat)

        # K = 1, 2, ..., MAX_OSC までの各最適化結果を比較します。
        if MAX_OSC is None:
            MAX_OSC = 5

        # AIC 最小の K が同一であれば、その K における
        # 誤差を計算します。
        if K_py == K_mat:
            K = K_py
            
            relative_errlist = list([[] * MAX_OSC])
            absolute_errlist = list([[] * MAX_OSC])
            # 比較する変数のリストです。
            var_list = [
                osc_param[K-1:K, 0:(2*J+1)*K+1],
                osc_AIC[0, K-1], 
                osc_mean[0:2*K, :, K-1], 
                osc_cov[0:2*K, 0:2*K, :, K-1], 
                osc_phase[0:K, :, K-1]]
            var_list_m = [
                osc_param_m[K-1:K, 0:(2*J+1)*K+1],
                osc_AIC_m[0, K-1], 
                osc_mean_m[0:2*K, :, K-1], 
                osc_cov_m[0:2*K, 0:2*K, :, K-1], 
                osc_phase_m[0:K, :, K-1]]

            name_list = [
                'osc_param', 'osc_AIC', 'osc_mean', 'osc_cov', 'osc_phase']

            # 変数を比較・誤差計算を行い、結果を表示します。
            print('Oscilator number K = ', K)
            for var, var_m, name in zip(var_list, var_list_m, name_list):
                err_relative, err_absolute = calculate_mean_square_error(
                    var, var_m)
                print('Square mean error of {}:'.format(name))
                print('relative {:.3e}:'.format(err_relative),
                    'absolute {:.3e}:'.format(err_absolute))
                relative_errlist[0].insert(K-1, np.array(err_relative))
                absolute_errlist[0].insert(K-1, np.array(err_relative))

            relative_errlist = np.array(relative_errlist)
            absolute_errlist = np.array(absolute_errlist)

            # AIC 最小の K における最大相対誤差が tol 以下であることをアサートします。
            assert (np.all(relative_errlist[0][K_py-1] <= tol))
            assert (np.all(absolute_errlist[0][K_py-1] <= tol))
    
    assert(
        str(e.value) == 'osc_decomp_uni: unknown algorithm'
        or
        str(e.value) == 'osc_decomp_multi: unknown algorithm')

