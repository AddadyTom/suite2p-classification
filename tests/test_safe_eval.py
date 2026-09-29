import pytest
import numpy as np
from fe_engine.fe_definitions import safe_eval_formula

def test_valid_math_formulas():
    # Setup dummy features namespace
    ns = {
        'solidity': np.array([0.5, 0.8, 1.0]),
        'mrs': np.array([2.0, 5.0, 10.0]),
        'npix': np.array([10.0, 20.0, 50.0])
    }
    
    # Test arithmetic operations
    res_add = safe_eval_formula("solidity + mrs", ns)
    assert np.allclose(res_add, ns['solidity'] + ns['mrs'])
    
    res_mul = safe_eval_formula("solidity * mrs", ns)
    assert np.allclose(res_mul, ns['solidity'] * ns['mrs'])
    
    res_sub_neg = safe_eval_formula("-solidity", ns)
    assert np.allclose(res_sub_neg, -ns['solidity'])
    
    res_pow = safe_eval_formula("mrs ** 2", ns)
    assert np.allclose(res_pow, ns['mrs'] ** 2)

    # Test math functions
    res_log = safe_eval_formula("log(solidity)", ns)
    assert np.allclose(res_log, np.log(ns['solidity']))
    
    res_exp = safe_eval_formula("exp(solidity)", ns)
    assert np.allclose(res_exp, np.exp(ns['solidity']))
    
    res_sqrt = safe_eval_formula("sqrt(npix)", ns)
    assert np.allclose(res_sqrt, np.sqrt(ns['npix']))

    res_complex = safe_eval_formula("log(solidity * mrs + 1)", ns)
    assert np.allclose(res_complex, np.log(ns['solidity'] * ns['mrs'] + 1))


def test_domain_boundaries():
    ns = {
        'x': np.array([0.0, -1.0, 5.0]),
        'y': np.array([0.0, 2.0, 0.0])
    }
    
    # Division by zero should be handled gracefully (denominator replaced by 1e-10)
    res_div = safe_eval_formula("x / y", ns)
    expected_div = ns['x'] / np.where(ns['y'] == 0, 1e-10, ns['y'])
    assert np.allclose(res_div, expected_div)
    
    # Log of zero/negative should be clipped to >= 1e-10
    res_log = safe_eval_formula("log(x)", ns)
    expected_log = np.log(np.maximum(ns['x'], 1e-10))
    assert np.allclose(res_log, expected_log)
    
    # Sqrt of negative should be clipped to >= 0.0
    res_sqrt = safe_eval_formula("sqrt(x)", ns)
    expected_sqrt = np.sqrt(np.maximum(ns['x'], 0.0))
    assert np.allclose(res_sqrt, expected_sqrt)


def test_safety_and_blacklist():
    ns = {
        'x': np.array([1.0, 2.0, 3.0])
    }
    
    # Disallowed variable names
    with pytest.raises(ValueError, match="Unknown feature name"):
        safe_eval_formula("x + y", ns)
        
    # Disallowed builtins/functions
    with pytest.raises(ValueError, match="Unsupported function"):
        safe_eval_formula("print(x)", ns)
        
    with pytest.raises(ValueError, match="Unsupported function"):
        safe_eval_formula("eval('x + 1')", ns)
        
    # Disallowed syntax structures (e.g. List comprehension, import statements, object attribute lookups)
    with pytest.raises(TypeError, match="Unsupported syntax expression structure"):
        safe_eval_formula("[val for val in x]", ns)
        
    with pytest.raises(TypeError, match="Unsupported syntax expression structure"):
        safe_eval_formula("x.__class__", ns)

    with pytest.raises(ValueError, match="Invalid mathematical syntax"):
        safe_eval_formula("import os", ns)

def test_case_insensitive_lookup():
    ns = {
        'snr': np.array([1.5, 2.5, 3.5]),
        'mrs': np.array([4.0, 5.0, 6.0])
    }
    # Should resolve SNR to snr, MRS to mrs
    res = safe_eval_formula("SNR * MRS", ns)
    assert np.allclose(res, ns['snr'] * ns['mrs'])
