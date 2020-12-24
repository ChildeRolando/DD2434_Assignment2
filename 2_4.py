""" This file is created as a template for question 2.4 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)! Previously, other students used NetworkX package to work with trees and graphs, keep in mind.

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you a single tree mixture (q2_4_tree_mixture).
    The mixture has 3 clusters, 5 nodes and 100 samples.
    We want you to run your EM algorithm and compare the real and inferred results
    in terms of Robinson-Foulds metric and the likelihoods.
    """
import numpy as np
import matplotlib.pyplot as plt


def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)


def em_algorithm(seed_val, samples, num_clusters, max_num_iter=20):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    This is a suggested template. Feel free to code however you want.
    """

    # Set the seed
    np.random.seed(seed_val)

    # TODO: Implement EM algorithm here.

    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    print("Running EM algorithm...")

    loglikelihood = []

    for iter_ in range(max_num_iter):
        loglikelihood.append(np.log((1 + iter_) / max_num_iter))

    from Tree import TreeMixture
    tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
    tm.simulate_pi(seed_val=seed_val)
    tm.simulate_trees(seed_val=seed_val)
    tm.sample_mixtures(num_samples=samples.shape[0], seed_val=seed_val)

    topology_list = []
    theta_list = []
    for i in range(num_clusters):
        topology_list.append(tm.clusters[i].get_topology_array())
        theta_list.append(tm.clusters[i].get_theta_array())

    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)
    # End: Example Code Segment

    for iter_ in  range(max_num_iter):
        # 1.calculate r(n,k)
        K = tm.num_clusters
        N = samples.shape[0]
        responsibilities = np.zeros(shape=(N, K))
        for k in range(K):
            for n in range(N):
                member = tm.pi[k]*tm.clusters[k].calculate_likelihood(samples[n])
                denonminator = sum([tm.pi[k]*tm.clusters[k].calculate_likelihood(samples[n]) for k in range(K)])
                responsibilities[n,k] = member/denonminator
        
        # 2.set pi`(k)
        for k in range(K):
            tm.pi[k] = np.average([responsibilities[n,k] for n in range(N)])
        
        # 3.1 calculate q(k)
        def q(k, s, a, t=None, b=None):
            if t is not None:
                member = sum([responsibilities[n,k] for n in range(N) if samples[n,s]==a and samples[n,t]==b])
            else:
                member = sum([responsibilities[n,k] for n in range(N) if samples[n,s]==a])
            denonminator = sum([responsibilities[n,k] for n in range(N)])
            return member/denonminator

        # 3.2 calculate Iqk(Xs,Xt)
        import math
        def I(k, s, t):
            return sum([sum([q(k, s, a, t, b)*math.log(q(k, s, a, t, b)/(q(k, s, a)*q(k, t, b))) if q(k, s, a, t, b) !=0 else 0 for a in [0,1]]) for b in [0,1]])
        for s in range(tm.num_nodes):
            print([I(0,s,t) for t in range(tm.num_nodes)])

        # 4 Kruskal and topology
        from Kruskal_v1 import Graph
        G = [Graph(tm.num_nodes) for _ in range(K)]
        for k in range(K):
            for u in range(tm.num_nodes):
                for v in range(u, tm.num_nodes):
                    G[k].addEdge(u, v, I(k, u, v))
        # calculate new topology
        for k in range(K):
            result = G[k].maximum_spanning_tree()
            topology_array = topology_list[k]
            visit_list = [0] #默认0为根节点，方便Θ的更新
            topology_array[visit_list[0]] = np.nan
            visited = []
            while len(visit_list):
                cur_node = visit_list[0]
                visit_list = visit_list[1:]
                for i,v in enumerate(result):
                    if i not in visited:
                        if v[0] == cur_node:
                            visit_list.append(v[1])
                            topology_array[v[1]] = cur_node
                            visited.append(i)
                        elif v[1] == cur_node:
                            visit_list.append(v[0])
                            topology_array[v[0]] = cur_node
                            visited.append(i)

        # 5.calculate Θk`(Xr)
        for k in range(K):
            for s in range(tm.num_nodes):
                if not np.isnan(topology_list[k][s]):
                    t = int(topology_list[k][s])
                    for a in [0,1]:
                        for b in [0,1]:
                            theta_list[k][s][a][b] = q(k,s,a,t,b)
                else:
                    for a in [0,1]:
                        theta_list[k][s][a] = q(k,s,a)
        # 6.refresh trees
        for k in range(k):
            tm.clusters[k].load_tree_from_direct_arrays(topology_list[k], theta_list[k])

        loglikelihood[iter_] = sum([math.log(sum([tm.pi[k]*tm.clusters[k].calculate_likelihood(samples[n]) for k in range(K)])) for n in range(N)])

    return loglikelihood, topology_list, theta_list


def main():
    print("Hello World!")
    print("This file demonstrates the flow of function templates of question 2.4.")

    seed_val = 123

    sample_filename = "data/q2_4/q2_4_tree_mixture.pkl_samples.txt"
    output_filename = "q2_4_results.txt"
    real_values_filename = "data/q2_4/q2_4_tree_mixture.pkl"
    num_clusters = 3

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array = em_algorithm(seed_val, samples, num_clusters=num_clusters)

    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, output_filename)

    for i in range(num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    if real_values_filename != "":
        print("\n4. Retrieve real results and compare.\n")
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        from Tree import Tree
        import dendropy
        tns = dendropy.TaxonNamespace()

        filename = "data/q2_4/q2_4_tree_mixture.pkl_tree_0_newick.txt"
        with open(filename, 'r') as input_file:
            newick_str = input_file.read()
        t0 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)

        filename = "data/q2_4/q2_4_tree_mixture.pkl_tree_1_newick.txt"
        with open(filename, 'r') as input_file:
            newick_str = input_file.read()
        t1 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)

        filename = "data/q2_4/q2_4_tree_mixture.pkl_tree_2_newick.txt"
        with open(filename, 'r') as input_file:
            newick_str = input_file.read()
        t2 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)

        filename = "q2_4_results.txt_em_topology.npy"  # This is the result you have.
        topology_list = np.load(filename)

        rt0 = Tree()
        rt0.load_tree_from_direct_arrays(topology_list[0])
        rt0 = dendropy.Tree.get(data=rt0.newick, schema="newick", taxon_namespace=tns)

        rt1 = Tree()
        rt1.load_tree_from_direct_arrays(topology_list[1])
        rt1 = dendropy.Tree.get(data=rt1.newick, schema="newick", taxon_namespace=tns)

        rt2 = Tree()
        rt2.load_tree_from_direct_arrays(topology_list[2])
        rt2 = dendropy.Tree.get(data=rt2.newick, schema="newick", taxon_namespace=tns)

        print("\tt0 vs inferred trees")
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt0))

        print("\tt1 vs inferred trees")
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt1))

        print("\tt2 vs inferred trees")
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt2))
        # TODO: Do RF Comparison

        print("\t4.2. Make the likelihood comparison.\n")
        # TODO: Do Likelihood Comparison
        from Tree import TreeMixture
        import math
        tm_true = TreeMixture(num_clusters, num_nodes)
        tm_true.load_mixture(real_values_filename)
        K = tm_true.num_clusters
        N = samples.shape[0]
        loglikelihood_R = sum([math.log(sum([tm_true.pi[k]*tm_true.clusters[k].calculate_likelihood(samples[n]) for k in range(K)])) for n in range(N)])
        print("real_loglikelihood:{}".format(-311))
        print("estimate_loglikelihood:{}".format(loglikelihood[-1]))
if __name__ == "__main__":
    main()
