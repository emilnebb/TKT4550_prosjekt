import numpy as np
import strid
from time import time
import functions as fun
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

class Cluster:

    def __init__(self, modes: dict,
                 ssid: strid.CovarianceDrivenStochasticSID,
                 linkage: str,
                 d_c: float,
                 distance_matrix: str,
                 power: float,
                 visualize = False):

        self.modes = modes
        self.ssid = ssid
        self.distance_matrix = distance_matrix
        self.d_c = d_c
        self.linkage = linkage
        self.visualize = visualize
        self.physical_coordinates = None
        self.mathematical_coordinates = None
        self.structural_modes_dict = None
        self.power = power


    def run_clustering(self):
        """
        Runs the three layer clustering model of k-means -> agglomerative
        -> k-means on the output from cov-SSI. Saves intermediate values
        to the cluster object for plotting, performance measurement, etc.

        """
        t0 = time()

        #Task 2.1 - Find relative difference
        difference = fun.rel_difference(self.modes, self.power)

        #Check relative difference integrity
        assert np.max(difference) <= 1, "Non-valid relative difference"

        #Task 2.1 - Using K-means clustering to separate all the poles into two groups
        kmeans = KMeans(n_clusters=2, random_state=0).fit(difference)
        labels1 = kmeans.labels_

        # Assign label to each mode
        physical_coordinates = []
        mathematical_coordinates = []
        count = 0
        physical_modes_dict = {}
        physical_modes_list = []
        num_modes = 0
        for order in range(49, 5, -1):
            modes_of_order = self.modes[order]
            # print("Order: " + str(order))
            physical_modes_in_order = []
            for mode in modes_of_order:
                mode.physical = labels1[count]
                if (mode.physical == 0):
                    physical_modes_in_order.append(mode)
                    physical_modes_list.append(mode)
                    physical_coordinates.append([mode.delta_frequency, mode.delta_damping, mode.delta_mac])
                else:
                    mathematical_coordinates.append([mode.delta_frequency, mode.delta_damping, mode.delta_mac])
                count += 1
            physical_modes_dict[order] = physical_modes_in_order
            num_modes += len(physical_modes_in_order)

        self.physical_coordinates = np.array(physical_coordinates)
        self.mathematical_coordinates = np.array(mathematical_coordinates)
        self.physical_modes_list = np.array(physical_modes_list)

        print("Number of physical modes = " + str(self.physical_coordinates.shape[0]))
        print("Number of mathematical modes = " + str(self.mathematical_coordinates.shape[0]))

        #Task 3.1 - Detect structural modes by hierarchical clustering

        #Start by computing distance matrix (3 options):
        if self.distance_matrix == "combined":
            distance = fun.distance_matrix(self.physical_modes_list)
        if self.distance_matrix == "mac":
            distance = fun.distance_matrix_mac(self.physical_modes_list)
        if self.distance_matrix == "eigen":
            distance = fun.distance_matrix_eigen(self.physical_modes_list)

        # Check the integrity of the distance matrix
        print("Check integrity of the distane matrix:")
        print("Max value = " + str(np.max(distance)))
        print("Mean value = " + str(np.mean(distance)))
        print("Min value = " + str(np.min(distance)))

        # Perform the actual clustering
        model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=self.linkage, distance_threshold=self.d_c)
        # The agglomerative clustering algorithm only finds 1 non-structural
        # mode with single linkage method. Works better so far with 'complete' or 'average'.

        y_hc = model.fit_predict(distance)
        model = model.fit(distance)
        #print((y_hc))
        print(np.max(y_hc))

        hierarchy = y_hc.tolist()

        num_modes_in_hierarchy = np.zeros((np.max(y_hc) + 1, 3))
        for i in range(0, len(num_modes_in_hierarchy)):
            num_modes_in_hierarchy[i, 0] = hierarchy.count(i)
            num_modes_in_hierarchy[i, 1] = hierarchy.count(i)
            num_modes_in_hierarchy[i, 2] = i

        # Assign hierarchy to each mode
        count = 0
        for order in range(49, 5, -1):
            modes_of_order = physical_modes_dict[order]
            for mode in modes_of_order:
                mode.cluster = hierarchy[count]
                count += 1

        #Task 4.1 - Using k-means to divide into one cluster with "many modes" (structural modes)
        # and one cluster with "few/scattered modes" (mathematcal modes).

        kmeans = KMeans(n_clusters=2, random_state=0).fit(num_modes_in_hierarchy[:, :2])
        labels2 = kmeans.labels_

        print("Number of clusters identified: " + str(len(labels2)))

        # Pick the label with lowest occurence
        labels2 = labels2.tolist()

        if (labels2.count(0) > labels2.count(1)):
            structural_label = 1
        else:
            structural_label = 0

        structural_hierarchies = []

        for i in range(0, len(labels2)):
            if (labels2[i] == structural_label):
                structural_hierarchies.append(i)

        # Assign hierarchy to each mode
        structural_modes_dict = {}
        for order in range(49, 5, -1):
            structural_modes_in_order = []
            for mode in physical_modes_dict[order]:
                if mode.cluster in structural_hierarchies:
                    mode.structural = 0
                    structural_modes_in_order.append(mode)
                else:
                    mode.structural = 1

            structural_modes_dict[order] = structural_modes_in_order

        self.structural_modes_dict = structural_modes_dict

        t1 = time()
        print("Clustering algorithm computational time = " + str(t1-t0) + " sec")


    def extract_modal_features(self, stabdiag: strid.stabdiag.StabilizationDiagram):
        """
        Extract the modal features of each detected mode as the average of all
        the components' features within each hierarchical cluster. Requires
        that run_clustering is called in advance such that self.structural_modes_dict
        is assigned a dictionry with modes.
        Parameters
        ----------
        stabdiag: object of class StabilizationDiagram. Dont actually need the object,
                but utilize the function filter_modes() within its class.

        Returns
            est_frequencies: 1a np.array with estimated frequencies
            est_damping: 1d np.array with estimated damping ratios
            est_modeshapes: 2d np.aray with estimated mode shapes
        -------

        """
        clustered_modes = fun.HierarchicalModes(stabdiag)

        modes_in_clusters = clustered_modes.clusters_dict(self.structural_modes_dict)

        hierarchies = sorted([*modes_in_clusters.keys()])
        clusters = defaultdict(list)

        for hierarchy in hierarchies:
            frequencies_in_hierarchy = []
            damping_in_hierarchy = []
            mode_shapes_in_hierarchy = []
            for element in modes_in_clusters[hierarchy]:
                frequencies_in_hierarchy.append(element.f)
                damping_in_hierarchy.append(element.xi)
                mode_shapes_in_hierarchy.append(element.v)

            f_mean = np.mean(np.array(frequencies_in_hierarchy))
            xi_mean = np.mean(np.array(damping_in_hierarchy))
            modal_shapes_mean = np.mean(np.array(mode_shapes_in_hierarchy), axis=0)

            clusters[hierarchy] = [f_mean, xi_mean, modal_shapes_mean]

        clustered_features = dict(clusters)

        # Sort the clusters by frequency to group into modes
        sorted_features_dict = {cluster: mode for cluster, mode in
                                sorted(clustered_features.items(), key=lambda item: item[1][0])}

        sorted_features_list = np.array(list(sorted_features_dict.values()))

        est_frequencies = sorted_features_list[:, 0]
        est_damping = sorted_features_list[:, 1]
        est_modeshapes = sorted_features_list[:, 2]

        return est_frequencies, est_damping, est_modeshapes