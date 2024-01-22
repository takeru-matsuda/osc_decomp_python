import pytest
import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
sys.path.append(current_dir)
from pyoscdecomp.pyoscdecompmulti.VARwithnoise_ll import VARwithnoise_ll
from pyoscdecomp.pyoscdecompmulti.whittle_multi_ll import whittle_multi_ll
from pyoscdecomp.pyoscdecompmulti.osc_multi_prof_ll import osc_multi_prof_ll
from pyoscdecomp.pyoscdecompmulti.mypolyeig import mypolyeig
from pyoscdecomp.pyoscdecompmulti.polyeig_VAR import polyeig_VAR
from pyoscdecomp.pyoscdecompmulti.whittle_multi_fit import whittle_multi_fit
from pyoscdecomp.pyoscdecompmulti.VAR_myule_ll import VAR_myule_ll
from pyoscdecomp.pyoscdecompmulti.VAR_myule import VAR_myule
from pyoscdecomp.pyoscdecompmulti.VAR_fit import VAR_fit
from pyoscdecomp.osc_ll_hess import osc_ll_hess
from check_eigenvector import check_eigenvector

# アサーションする桁数です。
decimal = 10
# 閾値です。
tol = np.float64(1.5 * 10**(-decimal))

# テスト データのディレクトリです。
matlab_out_dir = parent_dir + '/matlab_test_outputs/'

def test_mypolyeig():
    """mypolyeig のテストです。
    """        
    filename = matlab_out_dir + "mypolyeig.mat"
    data = scipy.io.loadmat(filename)
    C = data['C']
    C[0, -1] = np.array([[1.0, 0.0], [0.0, 1.0]])
    V_m = np.array(data['V'])
    z_m = np.array(data['z']).squeeze()
    s_m = np.array(data['s'])
    if_calculate_eigenvector = True
    if_calculate_cond = True
    V, z, s = mypolyeig(
        C, 
        if_calculate_cond=if_calculate_cond, 
        if_calculate_eigenvector=if_calculate_eigenvector)

    res = check_eigenvector(C, V, z).flatten()
    #print('POLYEIG RESIDUE = ', res, np.max(np.abs(res)))
    py_array = np.block([z, s.flatten()])
    mat_array = np.block([z_m, s_m.flatten()])
    diff_array = np.block([py_array - mat_array, res])
    assert (np.max(np.abs(diff_array)) <= tol)

def test_VARwithnoise_ll():
    """VARwithnoise_ll のテストです。
    """    
    filename = matlab_out_dir + "VARwithnoise_ll.mat"
    data = scipy.io.loadmat(filename)
    Y = np.array(data['Y'])
    A = np.array(data['A'])
    E = np.array(data['E'])
    R = np.array(data['R'])
    mll_m = np.array(data['mll']).item()

    mll = VARwithnoise_ll(Y, A, E, R)

    diff_array = mll - mll_m
    assert (np.max(np.abs(diff_array)) <= tol)


def test_whittle_multi_ll():
    """whittle_multi_ll のテストです。
    """    
    filename = matlab_out_dir + "whittle_multi_ll.mat"
    data = scipy.io.loadmat(filename)
    X = np.array(data['X'])
    y = np.array(data['y'])
    logbeta = np.array(data['logbeta'])
    mll_m = np.array(data['mll']).item()
    mll = whittle_multi_ll(X, y, logbeta)

    diff_array = mll - mll_m
    assert (np.max(np.abs(diff_array)) <= tol)

def test_osc_multi_prof_ll():
    """osc_multi_prof_ll のテストです。
    """    
    filename = matlab_out_dir + "osc_multi_prof_ll.mat"
    data = scipy.io.loadmat(filename)
    Y = np.array(data['Y'])
    param = np.array(data['param']).squeeze()
    init_theta = np.array(data['init_theta'])
    mll_m = np.array(data['mll']).item()
    grad_m = np.array(data['grad'])
    osc_tau2_m = np.array(data['osc_tau2']).item()

    mll, grad, osc_tau2 = osc_multi_prof_ll(Y, param, init_theta, True)

    py_array = np.block([mll, grad.flatten(), osc_tau2])
    mat_array = np.block([mll_m, grad_m.flatten(), osc_tau2_m])
    diff_array = py_array - mat_array
    assert (np.max(np.abs(diff_array)) <= tol)



def test_polyeig_VAR():
    """polyeig_VAR のテストです。
    """    
    filename = matlab_out_dir + 'polyeig_VAR.mat'
    data = scipy.io.loadmat(filename)
    A = np.array(data['inputA'])
    V_m = np.array(data['Vtmp'])
    z_m = np.array(data['tmp']).squeeze()

    V, z = polyeig_VAR(A)

    d = A.shape[0]    
    ARdeg = A.shape[1] // d
    C = list([[] * (ARdeg + 1)])
    for k in range(ARdeg):
        C[0].insert(k, -A[:, (ARdeg-k-1)*d:(ARdeg-k)*d])
    C[0].insert(ARdeg, np.eye(d))

    res = check_eigenvector(C, V, z).flatten()
    diff_array = np.block([z - z_m, res])
    assert (np.max(np.abs(diff_array)) <= tol)

def test_whittle_multi_fit():
    """whittle_multi_fit のテストです。
    """    
    filename = matlab_out_dir + 'whittle_multi_fit.mat'
    data = scipy.io.loadmat(filename)
    P = np.array(data['P'])
    p = np.array(data['p'])
    weight_m = np.array(data['weight'])
    weight = whittle_multi_fit(P, p)
    diff_array = weight - weight_m
    assert (np.max(np.abs(diff_array)) <= tol)


def test_VAR_myule_ll():
    """VAR_myule_ll のテストです。
    """    
    filename = matlab_out_dir + 'VAR_myule_ll.mat'
    data = scipy.io.loadmat(filename)
    Y = np.array(data['Y'])
    r = np.array(data['r'])
    C = np.array(data['C'])
    c = np.array(data['c'])
    mll_m = np.array(data['mll']).item()
    A_m = np.array(data['A'])
    E_m = np.array(data['E']).squeeze()

    mll, A, E = VAR_myule_ll(Y, r, C, c)

    py_array = np.block([mll, A.flatten(), E.flatten()])
    mat_array = np.block([mll_m, A_m.flatten(), E_m.flatten()])
    diff_array = py_array - mat_array
    assert (np.max(np.abs(diff_array)) <= tol)

def test_VAR_myule():
    """VAR_myule のテストです。
    """    
    filename = matlab_out_dir + 'VAR_myule.mat'
    data = scipy.io.loadmat(filename)
    Y = np.array(data['Y'])
    ARdeg = data['ARdeg'].item()
    A_m = np.array(data['A'])
    E_m = np.array(data['E'])
    r_m = np.array(data['r']).item()
    mll_m = np.array(data['mll']).item()

    A, E, r, mll = VAR_myule(Y, ARdeg)

    py_array = np.block([A.flatten(), E.flatten(), r, mll])
    mat_array = np.block([A_m.flatten(), E_m.flatten(), r_m, mll_m])

    diff_array = py_array - mat_array
    assert (np.max(np.abs(diff_array)) <= tol)

def test_VAR_fit():
    """VAR_fit のテストです。
    """    
    filename = matlab_out_dir + 'VAR_fit.mat'
    data = scipy.io.loadmat(filename)
    Y = np.array(data['Y'])
    MAX_VAR = data['MAX_VAR'].item()
    A_m = np.array(data['VARwithnoise_A'])
    E_m = np.array(data['VARwithnoise_E'])
    r_m = np.array(data['VARwithnoise_r'])
    AIC_m = np.array(data['VARwithnoise_AIC'])

    A, E, r, AIC = VAR_fit(Y, MAX_VAR)

    py_array = np.block([A.flatten(), E.flatten(), r.flatten(), AIC.flatten()])
    mat_array = np.block([A_m.flatten(), E_m.flatten(), r_m.flatten(), AIC_m.flatten()])

    diff_array = py_array - mat_array
    assert (np.max(np.abs(diff_array)) <= tol)


def test_osc_ll_hess():
    """osc_ll_hess のテストです。
    """    
    filename = matlab_out_dir + 'osc_ll_hess_multi.mat'
    data = scipy.io.loadmat(filename)
    y = np.array(data['Y'])
    fs = np.array(data['fs']).item()
    param = np.array(data['param'])
    hess_m = np.array(data['hess'])
    grad_m = np.array(data['grad'])
    mll_m = np.array(data['mll']).item()

    [hess, grad, mll] = osc_ll_hess(y, fs, param)

    py_array = np.block([hess.flatten(), grad.flatten(), mll])
    mat_array = np.block([hess_m.flatten(), grad_m.flatten(), mll_m])

    diff_array = py_array - mat_array
    # 相対誤差の最大値を表示します。    
    print(np.max(np.abs(diff_array) / np.abs(mat_array)))

    assert (np.max(np.abs(diff_array)) <= tol)

