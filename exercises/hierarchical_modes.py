import numpy as np
import strid.utils as utils
import strid.stabdiag as stab
from collections import defaultdict

class HierarchicalModes:

    def __init__(self, stabdiag: stab.StabilizationDiagram):
        self.stabdiag = stabdiag

    def clusters_dict(self, modes: dict) -> dict:
        """Plot modes in the stabilization diagram

        This method takes in a dict where the key is the
        order of the model and the value is a list of modes
        pertaining to the model order. No stability is checked
        in this method.
        Each mode is labeled with a color corresponding to its
        designated cluster. This method will only work if the number
        of clusters are less than or equal to the length of the color
        list, which is 302. Can be solved for an arbitrary number
        of clusters with modulus, resulting in dublicates of cluster
        colors.

        Arguments
        ---------
        modes : dict
            Dictionary where the key is the model order and
            the value is a list of strid.Mode instances.

        See Also
        --------
        filter_modes
            Method to filter out modes not relevant for further analysis and
            thus not plotted in stabilization diagram.
        find_stable_modes
            Method to classify stable modes.
        """
        #Todo: change method description

        clustered_modes = defaultdict(list)

        filtered_modes = self.stabdiag.filter_modes(modes) #this removes modes with negative frequencies
        orders = sorted([*modes.keys()])
        for order in orders:
            for mode in modes[order]:
                if mode not in filtered_modes[order]:
                    continue
                clustered_modes[mode.cluster].append(mode)

        return dict(clustered_modes)

