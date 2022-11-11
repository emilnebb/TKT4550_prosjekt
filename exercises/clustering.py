import numpy as np
import strid
import matplotlib.pyplot as plt
import scipy.signal
import functions as fun
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# Plotting dendogram: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
# https://www.youtube.com/watch?v=v7oLMvcxgFY&ab_channel=KindsonTheTechPro

class Cluster:

    def __init__(self, modes: dict,
                 ssid: strid.CovarianceDrivenStochasticSID,
                 linkage: str,
                 distance_matrix = "combined",
                 d_c = 0.04,
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


    def run_clustering(self):

        #Task 2.1 - Find relative difference
        difference = fun.rel_difference(self.modes)

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
        print("Chck integrity of the distane matrix:")
        print("Max value = " + str(np.max(distance)))
        print("Mean value = " + str(np.mean(distance)))
        print("Min value = " + str(np.min(distance)))

        # Perform the actual clustering
        model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=self.linkage, distance_threshold=self.d_c)
        # The agglomerative clustering algorithm only finds 1 non-structural
        # mode with single linkage method. Works better so far with 'complete' or 'average'.

        y_hc = model.fit_predict(distance)
        model = model.fit(distance)
        print((y_hc))
        print(np.max(y_hc))

        hierarchy = y_hc.tolist()

        num_modes_in_hierarchy = np.zeros((np.max(y_hc) + 1, 3))
        for i in range(0, len(num_modes_in_hierarchy)):
            num_modes_in_hierarchy[i, 0] = hierarchy.count(i)
            num_modes_in_hierarchy[i, 1] = hierarchy.count(i)
            num_modes_in_hierarchy[i, 2] = i

        structural_coordinates = []
        non_structural_coordinates = []

        for i in range(0, len(y_hc)):
            if y_hc[i] == 1:
                structural_coordinates.append(self.physical_coordinates[i, :])
            else:
                non_structural_coordinates.append(self.physical_coordinates[i, :])

        structural_coordinates = np.array(structural_coordinates)
        non_structural_coordinates = np.array(non_structural_coordinates)

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
        structural_label = None
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
            modes_of_order = physical_modes_dict[order]
            structural_modes_in_order = []
            for mode in modes_of_order:
                # print(mode.f)
                if mode.cluster in structural_hierarchies:
                    structural_modes_in_order.append(mode)

            structural_modes_dict[order] = structural_modes_in_order

        self.structural_modes_dict = structural_modes_dict

        # Making a new stabilization diagram with the structural modes
        stabdiag = strid.StabilizationDiagram()
        stabdiag.plot_clusters(self.structural_modes_dict)

        f, psd = self.ssid.psdy(nperseg=2 ** 10)

        stabdiag.axes_psd.semilogy(f, np.trace(np.abs(psd)), color=(0., 0., 0., .5), lw=.3)

        #Task 4.2 - Extract modal features of each detected mode as the average of all
        # the components’ features within each hierarchical cluster.

        clustered_modes = fun.HierarchicalModes(stabdiag)

        modes_in_clusters = clustered_modes.clusters_dict(structural_modes_dict)

        # Verfying numbers of clusters up against stabilization diagram
        assert len(modes_in_clusters) == 8, "Could not identify the correct number of structural modes"
        hierarchies = (list(modes_in_clusters.values()))

        est_frequencies = np.zeros(len(hierarchies))
        est_damping = np.zeros(len(hierarchies))
        est_modal_shapes = np.zeros((len(hierarchies), len(hierarchies)))

        for hier in hierarchies:
            frequencies_in_hierarchy = []
            damping_in_hierarchy = []
            mode_shapes_in_hierarchy = []
            for element in hier:
                frequencies_in_hierarchy.append(element.f)
                damping_in_hierarchy.append(element.xi)
                mode_shapes_in_hierarchy.append(element.v)

            est_frequencies[hierarchies.index(hier)] = np.mean(np.array(frequencies_in_hierarchy))
            est_damping[hierarchies.index(hier)] = np.mean(np.array(damping_in_hierarchy))
            est_modal_shapes[:, hierarchies.index(hier)] = np.mean(np.array(mode_shapes_in_hierarchy), axis=0)

        est_frequencies = np.sort(est_frequencies)
        est_damping = np.sort(est_damping)
        #TODO: modal shapes needs to be sorted in order to be compared with the correct ground truth

        return est_frequencies, est_damping, est_modal_shapes