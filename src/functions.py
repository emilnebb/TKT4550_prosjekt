import numpy as np
import strid
from time import time
import strid.stabdiag as stab
from collections import defaultdict

class HierarchicalModes:

    def __init__(self, stabdiag: stab.StabilizationDiagram):
        self.stabdiag = stabdiag

    def clusters_dict(self, modes: dict) -> dict:
        """Creates a new dictionary of modes where the key is the cluster
        they are assigned to in the hierarchical clustering. For each key
        in the dictionary, there will be a list of modes that are included
        in that cluster.

        Arguments
        ---------
        modes : dict
            Dictionary where the key is the model order and
            the value is a list of strid.Mode instances.

        Returns : dict
            Dictionary where the key is the cluster the mode
            are assigned to in the hierarchical clustering.
        """

        clustered_modes = defaultdict(list)

        filtered_modes = self.stabdiag.filter_modes(modes) #this removes modes with negative frequencies
        orders = sorted([*modes.keys()])
        for order in orders:
            for mode in modes[order]:
                if mode not in filtered_modes[order]:
                    continue
                clustered_modes[mode.cluster].append(mode)

        return dict(clustered_modes)

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

    ssid = strid.CovarianceDrivenStochasticSID(y, fs)

    modes = {}
    for i, order in enumerate(range(5, 50, 1)):
        A, C, G, R0 = ssid.perform(order, 25)
        modes[order] = strid.Mode.find_modes_from_ss(A, C, ssid.fs)

    return ssid, modes

def rel_diff_freq(f1, f2):
    "Find relative difference between two different frequencies"
    return np.abs(f2-f1)/(np.max([np.abs(f1), np.abs(f2)]))

def find_nearest_neighbour(home_mode: strid.Mode, potential_neighbours: list) -> strid.Mode:
    """
    Find nearest neighbour to the home mode. Uses relative distance between frequencies as measure.
    Parameters
    ----------
    home_mode: the mode we want to find the nearest neighbour to, type Mode
    potential_neighbours: list of modes with candidate of neighbours

    Returns
    -------
    nearest neighbour,type Mode
    """

    neighbour = potential_neighbours[0]

    for mode in range(1, len(potential_neighbours)):
        if (rel_diff_freq(home_mode.f, potential_neighbours[mode].f) <
        rel_diff_freq(home_mode.f, neighbour.f)):
            neighbour = potential_neighbours[mode]

    return neighbour


def rel_difference(modes: dict, power: float) -> np.ndarray:
    """
    Finding the relative difference in frequency, damping and mode shape between
    a mode and its nearest neighbour.
    Parameters
    ----------
    modes: a dictionary with modes by order

    Returns: a nx3 ndarray with relative difference in frequency, damping a nd mode shape for each mode.
    -------
    """

    cluster_meat = []
    num_modes = 0

    for order in range(49, 5, -1):
        modes_of_order = modes[order]
        # print("Order: " + str(order))
        num_modes += len(modes_of_order)
        counter = 0
        for mode in modes_of_order:
            counter += 1
            neighbour = find_nearest_neighbour(mode, modes[order - 1])

            if (np.isnan(rel_diff_freq(neighbour.f, mode.f))):
                mode.delta_frequency = 1
            else:
                mode.delta_frequency = rel_diff_freq(neighbour.f, mode.f)**power

            if (np.isnan(np.abs(neighbour.xi - mode.xi))):
                mode.delta_damping = 1
            else:
                mode.delta_damping = (np.abs(neighbour.xi - mode.xi) / np.max([np.abs(neighbour.xi), np.abs(mode.xi)]))**power

            if (np.isnan(strid.utils.modal_assurance_criterion(neighbour.v, mode.v))):
                mode.mac = 1
            else:
                mode.delta_mac = (1 - strid.utils.modal_assurance_criterion(neighbour.v, mode.v))**power

            cluster_meat.append([mode.delta_frequency, mode.delta_damping, mode.delta_mac])

    return np.array(cluster_meat)



def distance_matrix(modes: np.ndarray) -> np.ndarray:
    """
    Computes the distance matrix between structural modes.
    The distance between two modes i and j is defined as:
    dc_ij = delta_eigenvalues_ij + (1 - MAC_ij)
    Parameters
    ----------
    modes: 1d numpy array with elements of class Mode

    Returns: 2d numpy array that is a distance matrix
    -------

    """
    #Meassuring the computational time
    t0 = time()
    #Preallocatig matrix to store the distances in
    dist_matrix = np.zeros((modes.shape[0], modes.shape[0]))

    for i in range(0, dist_matrix.shape[0]):
        for j in range(0, dist_matrix.shape[1]):
            # Computing the distance at each element
            eigen_i = np.sqrt(modes[i].eigenvalue ** 2)
            eigen_j = np.sqrt(modes[j].eigenvalue ** 2)
            if i != j:
                dist_matrix[i, j] = (np.abs((eigen_i - eigen_j)) / np.max([eigen_i, eigen_j])) + (
                            1 - strid.modal_assurance_criterion(modes[i].v, modes[j].v))

    t1 = time()
    print("Distance matrix computational time = " + str(t1-t0) + " sec")

    return dist_matrix

def distance_matrix_mac(modes: np.ndarray) -> np.ndarray:
    """
    Computes the distance matrix between structural modes.
    The distance between two modes i and j is defined as:
    dc_ij =  1 - MAC_ij
    As proposed in "Application of OMA to operational wind turbine"
    by Tcherniak et.Al. page 6.
    Parameters
    ----------
    modes: 1d numpy array with elements of class Mode

    Returns: 2d numpy array that is a distance matrix
    -------

    """
    #Meassuring the computational time
    t0 = time()
    #Preallocatig matrix to store the distances in
    dist_matrix = np.zeros((modes.shape[0], modes.shape[0]))

    for i in range(0, dist_matrix.shape[0]):
        for j in range(0, dist_matrix.shape[1]):
            # Computing the distance at each element
            if i != j:
                dist_matrix[i, j] = (1 - strid.modal_assurance_criterion(modes[i].v, modes[j].v))

    t1 = time()
    print("Distance matrix computational time = " + str(t1-t0) + "sec")

    return dist_matrix