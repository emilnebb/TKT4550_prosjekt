import numpy as np
import strid

def covSSI(path: str) -> (strid.CovarianceDrivenStochasticSID, dict):
    """
    Uses covariance driven stochastic subspace identification to perform system identification.
    Based on example notebook "01a-stochastic-system-identification-CovSSI.ipynb"
    Parameters
    ----------
    path: str of path to stored data.

    Returns: a CovarianceDrivenStochasticSID object and a dictionary with modes.
    -------

    """

    data = np.load(path)
    y = data["y"]
    fs = data["fs"]
    true_f = data["true_frequencies"]
    true_xi = data["true_damping"]
    true_modeshapes = data["true_modeshapes"]

    ssid = strid.CovarianceDrivenStochasticSID(y, fs)

    modes = {}
    for i, order in enumerate(range(5, 50, 1)):
        A, C, G, R0 = ssid.perform(order, 25)
        modes[order] = strid.Mode.find_modes_from_ss(A, C, ssid.fs)

    return ssid, modes