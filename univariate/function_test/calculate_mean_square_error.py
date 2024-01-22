import numpy as np

def calculate_mean_square_error(var1, var2):
    """
    Calcuate mean square error of two ndarray variables.
    Args:
        var1: ndarray
        var2: ndarray
    Returns:
        err: float
             Square root of mean square error of var1 relative to var2.
    Raises:
        ValueError: either var1 or var2 is not ndarray object.
        ValueError: var1 and var2 have different shape.
    """    

    is_ndarray = (isinstance(var1, type(np.zeros(1))) 
                  and isinstance(var2, type(np.zeros(1))))
    is_float = (isinstance(var1, float) 
                  and isinstance(var2, float))

    if not (is_ndarray or is_float):
        raise ValueError(
            'compare_variables: var1 or var2 is not ndarray nor float.')

    if is_ndarray:
        if (var1.shape != var2.shape):
            raise ValueError(
                'compare_variables: var1 and var2 have different shape.')

        err_relative = np.sqrt(
            np.square(var1 - var2).mean() 
            / np.square(var2).mean())

        err_absolute = np.sqrt(
            np.square(var1 - var2).mean())
            
    elif is_float:
        err_relative = np.abs(var1 - var2) / np.abs(var2)
        err_absolute = np.abs(var1 - var2)

    return err_relative, err_absolute
    
