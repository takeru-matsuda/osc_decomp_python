import pytest
import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
sys.path.append(current_dir)
from pyoscdecomp.pyoscdecompuni.whittle_uni_ll import whittle_uni_ll
from pyoscdecomp.pyoscdecompuni.whittle_uni_fit import whittle_uni_fit
from pyoscdecomp.pyoscdecompuni.ARwithnoise_ll import ARwithnoise_ll
from pyoscdecomp.pyoscdecompuni.osc_uni_prof_ll import osc_uni_prof_ll
from pyoscdecomp.pyoscdecompuni.AR_ll import AR_ll
from pyoscdecomp.pyoscdecompuni.AR_fit import AR_fit
from pyoscdecomp.pyoscdecompuni.armyule import armyule
from pyoscdecomp.osc_smooth import osc_smooth
from pyoscdecomp.osc_ll_hess import osc_ll_hess
from calculate_mean_square_error import calculate_mean_square_error

# アサーションする桁数です。
decimal = 6

def test_whittle_uni_ll():
    """whittle_uni_ll のテストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/whittle_uni_ll.mat'
    data = scipy.io.loadmat(filename)
    X = np.array(data['X'])
    y = np.array(data['y'])
    b = np.array(data['b'])
    mll_m = np.array(data['mll']).item()

    mll = whittle_uni_ll(X, y, b)

    np.testing.assert_array_almost_equal(mll, mll_m, decimal=decimal)


def test_ARwithnoise_ll():
    """ARwithnoise_ll のテストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/ARwithnoise_ll.mat'
    data = scipy.io.loadmat(filename)
    y = np.array(data['y'])
    param = np.array(data['param'])
    mll_m = np.array(data['mll']).item()
    R_m = np.array(data['R']).item()

    mll, R = ARwithnoise_ll(y, param)

    py_array = np.array([mll, R])
    mat_array = np.array([mll_m, R_m])

    np.testing.assert_array_almost_equal(py_array, mat_array, decimal=decimal)


def test_AR_ll():
    """AR_ll のテストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/AR_ll.mat'
    data = scipy.io.loadmat(filename)
    y = np.array(data['y'])
    p = (np.array(data['p'])).reshape((1, 20))

    ar_ll_m = np.array(data['ar_ll']).item()
    ehat_m = np.array(data['ehat']).item()
    
    [ar_ll, ehat] = AR_ll(y, p)
    py_array = np.array([ar_ll, ehat])
    mat_array = np.array([ar_ll_m, ehat_m])

    np.testing.assert_array_almost_equal(py_array, mat_array, decimal=decimal)


def test_osc_uni_prof_ll():
    """osc_uni_prof_ll テストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/osc_uni_prof_ll.mat'
    data = scipy.io.loadmat(filename)
    y = np.array(data['y'])
    param = np.array(data['param'])
    init_theta = np.array(data['init_theta'])
    mll_m = np.array(data['mll']).item()
    grad_m = np.array(data['gradient'])
    osc_tau2_m = np.array(data['osc_tau2']).item()

    [mll, grad, osc_tau2] = osc_uni_prof_ll(
        y, param.squeeze(), init_theta, True)
    py_array = np.block([mll, grad.squeeze(), osc_tau2])
    mat_array = np.block([mll_m, grad_m.squeeze(), osc_tau2_m])
    np.testing.assert_array_almost_equal(py_array, mat_array, decimal=decimal)



def test_osc_smooth():
    """osc_smooth のテストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/osc_smooth.mat'
    data = scipy.io.loadmat(filename)
    y = np.array(data['y'])
    # print('y shape in osc_smooth= ', y.shape)
    fs = np.array(data['fs']).item()
    param = np.array(data['param'])
    x_smooth_m = np.array(data['x_smooth'])
    V_smooth_m = np.array(data['V_smooth'])

    [x_smooth, V_smooth] = osc_smooth(y, fs, param)

    py_array = np.block([x_smooth.flatten(), V_smooth.flatten()])
    mat_array = np.block([x_smooth_m.flatten(), V_smooth_m.flatten()])
    np.testing.assert_array_almost_equal(py_array, mat_array, decimal=decimal)



def test_osc_ll_hess():
    """osc_ll_hess のテストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/osc_ll_hess_uni.mat'
    data = scipy.io.loadmat(filename)
    y = np.array(data['y'])
    # print('y shape in osc_ll_hess= ', y.shape)
    fs = np.array(data['fs']).item()
    param = np.array(data['osc_param_in'])
    hess_m = np.array(data['hess'])
    grad_m = np.array(data['grad'])
    mll_m = np.array(data['mll']).item()

    [hess, grad, mll] = osc_ll_hess(y, fs, param)


    py_array = np.block([hess.flatten(), grad.flatten(), mll])
    mat_array = np.block([hess_m.flatten(), grad_m.flatten(), mll_m])
    np.testing.assert_array_almost_equal(py_array, mat_array, decimal=decimal)


def test_armyule():
    """armyule のテストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/armyule.mat'
    data = scipy.io.loadmat(filename)
    y = np.array(data['y'])
    ARdeg = data['ARdeg'].item()

    A_m = np.array(data['A'])
    E_m = np.array(data['E']).item()
    R_m = np.array(data['R']).item()

    A, E, R = armyule(y, ARdeg)

    py_array = np.block([A.flatten(), E, R])
    mat_array = np.block([A_m.flatten(), E_m, R_m])
    np.testing.assert_array_almost_equal(py_array, mat_array, decimal=decimal)


def test_AR_fit():
    """AR_fit のテストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/AR_fit.mat'
    data = scipy.io.loadmat(filename)
    y = np.array(data['y'])
    MAX_AR = data['MAX_AR'].item()
    ARwithnoise_param_m = np.array(data['ARwithnoise_param'])
    ARwithnoise_AIC_m = np.array(data['ARwithnoise_AIC'])
    AR_param_m = np.array(data['ar_param'])
    AR_AIC_m = np.array(data['ar_aic'])

    [ARwithnoise_param, 
     ARwithnoise_AIC, 
     AR_param, 
     AR_AIC] = AR_fit(y, MAX_AR)


    py_array = np.block([ARwithnoise_param.flatten(), 
                         ARwithnoise_AIC.flatten(), 
                         AR_param.flatten(), 
                         AR_AIC.flatten()])
    mat_array = np.block([ARwithnoise_param_m.flatten(), 
                         ARwithnoise_AIC_m.flatten(), 
                         AR_param_m.flatten(), 
                         AR_AIC_m.flatten()])
    np.testing.assert_array_almost_equal(py_array, mat_array, decimal=decimal)

def test_whittle_uni_fit():
    """whittle_uni_fit のテストです。
    """    
    filename = parent_dir + '/matlab_test_outputs/whittle_uni_fit.mat'
    data = scipy.io.loadmat(filename)
    P = np.array(data['P'])
    p = np.array(data['p'])
    weight_m = np.array(data['weight'])

    weight = whittle_uni_fit(P, p)

    np.testing.assert_array_almost_equal(weight, weight_m, decimal=decimal)
