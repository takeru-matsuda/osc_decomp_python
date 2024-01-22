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


# osc_decomp のテスト ケースです。
@pytest.mark.parametrize('sizeY1, MAX_OSC, MAX_AR, algorithm, grad, matlab_file', [
    (1, 6, 21, 'quasi-newton', True, 'matlab_test_outputs/osc_decomp_1.mat'),
    (1, 6, 21, 'trust-region', True, 'matlab_test_outputs/osc_decomp_2.mat'),
    (1, 6, 21, 'quasi-newton', False, 'matlab_test_outputs/osc_decomp_3.mat'),
    (1, 6, 21, None, None, 'matlab_test_outputs/osc_decomp_4.mat'),
    (1, 6, None, None, None, 'matlab_test_outputs/osc_decomp_5.mat'),
    (1, None, None, None, None, 'matlab_test_outputs/osc_decomp_6.mat'),
    (5, None, None, None, None, 'matlab_test_outputs/osc_decomp_6.mat'),  # 複合時系列(未対応)です。
    (1, None, None, 'Nelder-Mead', None, 'matlab_test_outputs/osc_decomp_6.mat')  # 未対応の最適化です。
])
def test_osc_decomp(sizeY1, MAX_OSC, MAX_AR, algorithm, grad, matlab_file):
    """
    osc_decomp のテストです。
    引数ごとにテストを行います。
    """    
    data = scipy.io.loadmat(
        parent_dir + '/matlab_test_outputs/osc_decomp_uniinputs.mat')
    y = np.array(data['Y'])
    Y = np.tile(y, (sizeY1, 1))
    print('Y.shape = ', Y.shape)
    fs = np.array(data['fs']).item()

    data = scipy.io.loadmat(parent_dir + '/' + matlab_file)
    osc_param_m = np.array(data['osc_param'])
    osc_AIC_m = np.array(data['osc_AIC'])
    osc_mean_m = np.array(data['osc_mean'])
    osc_cov_m = np.array(data['osc_cov'])
    osc_phase_m = np.array(data['osc_phase'])

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

        # 比較する変数のリストです。
        var_list = [
            osc_param, osc_AIC, osc_mean, osc_cov, osc_phase]
        var_list_m = [
            osc_param_m, osc_AIC_m, osc_mean_m, osc_cov_m, osc_phase_m]
        name_list = [
            'osc_param', 'osc_AIC', 'osc_mean', 'osc_cov', 'osc_phase']

        # 変数を比較・誤差計算を行い、結果を表示します。
        relative_errlist = []
        absolute_errlist = []
        for var, var_m, name in zip(var_list, var_list_m, name_list):
            err_relative, err_absolute = calculate_mean_square_error(var, var_m)
            print('Square mean error of {}:'.format(name))
            print('relative {:.3e}:'.format(err_relative),
                'absolute {:.3e}:'.format(err_absolute))
            relative_errlist.append(err_relative)
            absolute_errlist.append(err_absolute)

        relative_errlist = np.array(relative_errlist)
        absolute_errlist = np.array(absolute_errlist)

        # アサーション テストとして、すべての誤差が閾値以下であるかアサートします。
        tol = 1e-5
        assert(np.all(relative_errlist <= tol))
        assert(np.all(absolute_errlist <= tol))
    
    assert(str(e.value) == 'osc_decomp_uni: unknown algorithm' or
           str(e.value) == 'osc_decom_multi is not implemented yet.')

