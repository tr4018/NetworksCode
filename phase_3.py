"""
    builds a graph based on a hybrid preferential attachment and random attachment.
    author - Talia Rahall, March 2021
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from scipy.stats import linregress, ks_2samp

import phase_1 as pa
import phase_2 as ra
from log_bin import logbin
from simple_graph import SimpleGraph


def theo_pdf(m, k, p) -> float:
    if k < m:
        return 0.0

    if p == 2.0 / 3.0:
        return 3 * float((2 * m) * (2 * m + 1) * (2 * m + 2)) / \
               ((m + k) * (m + k + 1) * (m + k + 2) * (m + k + 3))
    elif p == 1.0 / 2.0:
        return 4 * float((3 * m) * (3 * m + 1) * (3 * m + 2) * (3 * m + 3)) / \
               ((2 * m + k) * (2 * m + k + 1) * (2 * m + k + 2) * (2 * m + k + 3) * (2 * m + k + 4))
    elif p == 1.0 / 3.0:
        return 6 * float((5 * m) * (5 * m + 1) * (5 * m + 2) * (5 * m + 3) * (5 * m + 4) * (5 * m + 5)) / \
               ((4 * m + k) * (4 * m + k + 1) * (4 * m + k + 2) * (4 * m + k + 3) * (4 * m + k + 4) *
                (4 * m + k + 5) * (4 * m + k + 6))
    else:
        raise Exception('probability ', str(p), ' not supported')


def theo_cdf(m, k, q) -> float:
    return sum([theo_pdf(m, i, q) for i in range(m, k + 1)])

def continuous_ppf(u, m, q):
    c = 2.0 / q * np.power(2*m-m*q, 2.0/q)
    z = 1.0/(2*m - m*q) - 2*u/c
    k = 1.0 / q * ((1.0/z)**(q/2) - 2*m*(1.0 - q))
    return k


def theo_ppf(m, q, qd) -> int:
    s = 0.0
    k = m
    while s < q:
        s += theo_pdf(m, k, qd)
        k += 1
    return k - 1


def build_graph_para(m0: int, m: int, ne: int, q: float) -> SimpleGraph:
    # m0 - initial number of vertices
    # m - additional number of vertices
    # ne - number of edges per vertex
    # q  - probability to choose between pa and ra

    if ne > m0:
        raise Exception('m0 must be greater than ne')

    print('**** preferential/random attachment ****')
    g = SimpleGraph()
    edges = []
    for i in range(m0):
        g.add_vertex()

    for source in range(m0 - 1):
        for iter_ in range(ne):
            target = rnd.randint(0, m0)
            if source != target and target not in g.get_neighbours(source):
                g.add_edge_slowly(source, target)
                edges.append((source, target))

    for source in range(m0, m0 + m):
        g.add_vertex()
        if rnd.uniform() < q:
            #  chose pa
            g, edges = pa.add_edge(g, source, ne, edges)
        else:
            #  chose ra
            g, edges = ra.add_edge(g, source, ne, edges)
    return g


def plot_for_qs():
    rnd.seed(12345)
    num_vertices = 200000
    ne = 2
    m0 = 3

    qs = [0., 0.5, 1.]

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    ax.set_title(r'prob vs k')
    ax.set_xlabel('k')
    ax.set_ylabel(r'$p_N(k)$')

    for q in qs:
        print('q =', q)
        g = build_graph_para(m0, num_vertices, ne, q)
        dd = g.get_degree_dist()
        data = pa.expand_counts(dd)
        x, prob = logbin(data, 1.2)
        idx = x >= ne
        x = x[idx]
        prob = prob[idx]
        x = np.log10(x)
        y = np.log10(prob)
        ax.plot(x, y, linestyle='None', marker='+', label='emp, q=' + str(q))
        res = linregress(x, y)
        print(res.slope, res.intercept)
        if q == qs[0]:
            ks = np.power(10.0, x)
            theo_ra = np.log10(np.array([ra.theo_pdf(ne, k) for k in ks]))
            ax.plot(x, theo_ra, label='theo ra')
        if q == qs[len(qs) - 1]:
            ks = np.power(10.0, x)
            theo_pa = np.log10(np.array([pa.theo_pdf(ne, k) for k in ks]))
            ax.plot(x, theo_pa, label='theo pa')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


def ks_gof_test():
    rnd.seed(12345)
    num_vertices = 100000
    ne = 2
    m0 = 5
    qs = [2.0 / 3.0, 1.0 / 2.0, 1.0 / 3.0]

    for q in qs:
        print(q)
        g = build_graph_para(m0, num_vertices, ne, q)
        dd = g.get_degree_dist()
        data = pa.expand_counts(dd)

        u = rnd.uniform(0, 1, 1000)
        theo_sample = np.array([theo_ppf(ne, qd, q) for qd in u])

        res = ks_2samp(data, theo_sample)
        print(res)
        print('dcrit', 1.36*np.sqrt(1/len(theo_sample)+1/len(data)))
        
        # con_sample = np.array([continuous_ppf(q, ne, qd) for qd in u])
        
        
        # res2 = ks_2samp(data, con_sample)
        # print(res2)


def degree_dist():
    rnd.seed(12345)
    num_vertices = 100000
    ne = 2
    m0 = 5
    num_paths=15
    qs = [2.0 / 3.0, 1.0 / 2.0, 1.0 / 3.0]
    fig = plt.figure(figsize=(18, 5))
   
    # ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    labels = [r'$q=2/3$',r'$q=1/2$',r'$q=1/3$']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    i=0
    labelsa = ['A', 'B', 'C', 'D', 'E', 'F']
    total_stds=[]
    xaxis=[]
    for q in qs:
        data = []
        

        for path in range(num_paths): 
            
            print('path=', path)
            g = build_graph_para(m0, num_vertices, ne, q)
            
            dd = g.get_degree_dist()
            data.extend(pa.expand_counts(dd))
        
        x, y = logbin(data, scale=1.2)
        xaxis.append(x)
        x=np.insert(x,0,0)
        counts, bins = np.histogram(data, x)
        indexing = np.cumsum(counts)
        vals = []
        
        data = np.sort(data)
        for j in range(len(indexing)):
            if j == 0:
                vals.append(np.std(data[:indexing[j]+1]))

            if j > 0:
                vals.append(np.std(data[indexing[j-1]+1:indexing[j]+1]))
                
        total_stds.append(vals)
        
        
        ax = plt.subplot(1,3,i+1)

        ax.set_xlabel(r'$k$', fontsize=21)
        ax.set_ylabel(r'$p(k)$', fontsize=21)

        print('q =', q)
        
        
        x=x[1:]
        prob=y
        
        idx = x >= ne
        x = x[idx]
        prob = prob[idx]
        theo_prob = np.log10([theo_pdf(ne, k, q) for k in x])
        if q == qs[0]:
            x_m = x + ne
        elif q == qs[1]:
            x_m = x + 2 * ne
        else:
            x_m = x + 4 * ne
        x = np.log10(x)
        y = np.log10(prob)
        x_m = np.log10(x_m)
        c_prob = continuous_prob(10**x, ne, q)

        ax.loglog(10**x, 10**y, label=labels[i], marker = 'o', color = colors[i], markersize= 6,markeredgewidth=0.4, markeredgecolor='black', zorder=2, linestyle='None')
        ax.loglog(10**x, 10**theo_prob, color='black', zorder=-1)
        ax.loglog(10**x, c_prob, color='black', linestyle='dashed', label='Cont. Approx.', zorder=-1)
        ax.text(0.90, 0.95, labelsa[i], transform=ax.transAxes,fontsize=16, fontweight='bold', va='top') 

        # res = linregress(x_m, y)
        # print(res.slope, res.intercept)
        ax.legend(loc='lower left', fontsize='x-large')
        # ax.plot(x_m, res.slope * x_m + res.intercept, label='regression')
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.show()
        i+=1
        

    plt.tight_layout()
    plt.show()


def qq_plot():

    rnd.seed(12345)
    num_vertices = 200000
    ne = 2
    q = 2.0/3.0

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel('theo quantiles')
    ax1.set_ylabel('sample quantiles')
    ax2.set_xlabel('k')
    ax2.set_ylabel('prob')

    quantiles = rnd.uniform(0.0, 1.0, size=20)
    print('num edges', ne)
    m0 = ne*(ne+1)
    g = build_graph_para(m0, num_vertices, ne, q)
    dd = g.get_degree_dist()
    # qq plot
    prob = np.array(dd) / sum(dd)
    print(sum(prob))
    theo_q = [theo_ppf(ne, qd, q) for qd in quantiles]
    print(theo_q)
    empirical_q = [pa.eppf(prob, qd) for qd in quantiles]
    print(empirical_q)
    ax1.set_title('q-q plot')
    ax1.plot(theo_q, empirical_q, '+')
    ax1.plot(theo_q, theo_q)
    # cum density plot
    k_s = np.linspace(0, 100, endpoint=True, num=100, dtype='int')
    theo_cums = [theo_cdf(ne, k, q) for k in k_s]
    emp_cums = [pa.ecdf(prob, k) for k in k_s]
    ax2.set_title('cumulative prob')
    ax2.plot(k_s, theo_cums)
    ax2.plot(k_s, emp_cums, '+')

    plt.show()


def continuous_prob(x, m, q):
    if q == 0:
        return 1.0 / m * np.exp(1.0 - x/m)
    else:
        c = 2.0 / q * np.power(2*m-m*q, 2.0/q)
        return c / np.power(2*m*(1.0-q) + q*x, 1.0 + 2.0/q)


def continuous_approx():
    rnd.seed(12345)
    num_vertices = 200000
    ne = 4
    m0 = 5

    qs = [0.0, 0.333, .5, 0.6667, 1]

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title(r'prob vs k')
    ax.set_xlabel('k')
    ax.set_ylabel(r'$p_N(k)$')

    for q in qs:

        print('q =', q)
        g = build_graph_para(m0, num_vertices, ne, q)
        dd = g.get_degree_dist()

        data = pa.expand_counts(dd)
        x, prob = logbin(data, 1.2)
        idx = x >= ne
        x = x[idx]
        prob = prob[idx]
        c_prob = continuous_prob(x, ne, q)
        x = np.log10(x)
        y = np.log10(prob)

        ax.plot(x, y, linestyle='None', marker='+', label='emp, q=' + str(q))
        ax.plot(x, np.log10(c_prob))

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    # plot_for_qs()
    # qq_plot()
    # degree_dist()
    ks_gof_test()
    # continuous_approx()
