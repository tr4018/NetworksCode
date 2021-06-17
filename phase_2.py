from math import ceil
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy.stats import linregress, ks_2samp

from simple_graph import SimpleGraph
from phase_1 import expand_counts, ecdf, eppf

from log_bin import logbin


def add_edge(g: SimpleGraph, source: int, ne: int, edges: []) -> ():
    n = g.get_num_vertices()
    for edge in range(ne):
        target = rnd.randint(0, n)
        if source != target and target not in g.get_neighbours(source):
            g.add_edge_slowly(source, target)
            edges.append((source, target))
    return g, edges


def build_graph_ra(m0: int, m: int, ne: int) -> SimpleGraph:

    # m0 - initial number of vertices
    # m - additional number of vertices
    # ne - number of edges per vertex

    print('**** random attachment ****')
    g = SimpleGraph()
    for i in range(m0):
        g.add_vertex()

    edges = []
    for source in range(m0 - 1):
        g, edges = add_edge(g, source, ne, edges)

    for source in range(m0, m0+m):
        g.add_vertex()
        g, edges = add_edge(g, source, ne, edges)
    return g


def theo_pdf(m, k) -> float:
    return 0.0 if k < m else 1.0/(m+1) * (float(m)/(float(m+1)))**(k-m)


def theo_cdf(m, k) -> float:
    #return 0.0 if k < m else sum([theo_pdf(m, i) for i in range(m, k+1)])
    return 0.0 if k < m else 1.0 - (float(m)/float(m+1))**(k+1-m)


def theo_ppf(m, q) -> int:
    # s = 0.0
    # k = m
    # while s < q:
    #     s += theo_pdf(m, k)
    #     k += 1
    # return k - 1

    if q == 1.0:
        q -= 1.0e-8
    alpha = float(m) / float(m+1)
    k = m - 1 + ceil(np.log(1.0 - q) / np.log(alpha))
    return k


# def ecdf(prob: np.array, k: int) -> float:
#     if k < 0:
#         return 0.0
#     elif k >= len(prob):
#         return 1.0
#     else:
#         return np.sum(prob[:k+1])
#
#
# def eppf(prob: np.array, q: float) -> int:
#     # empirical percentage point function
#     s = 0.0
#     k = 0
#     while s < q:
#         if k == len(prob):
#             break
#         s += prob[k]
#         k += 1
#     return k - 1


def degree_dist():
    rnd.seed(12345)
    num_vertices = 100000
    num_edges = [2, 4, 8, 16, 32, 64]
    m0 = [num * 2 + 1 for num in num_edges]
    num_paths = 20

    fig, ax = plt.subplots(1,1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    mylegend=[r'$m=2$', r'$m=4$', r'$m=8$', r'$m=16$', r'$m=32$', r'$m=64$']
    mi = 0
    xaxis=[]
    total_stds=[]

    for ne in num_edges:
        data = []
  
        for path in range(num_paths):
            
            print('num edges', ne, 'path=', path)
            g = build_graph_ra(m0[mi], num_vertices, ne)    
            dd = g.get_degree_dist()
            data.extend(expand_counts(dd))
            
            
            
        x, y = logbin(data, scale=1.3)
        # if ne in (2,4,8):
        #     x=x[4:]
        #     y=y[4:]
        # if ne in (16, 32,64):
        #     x=x[10:]
        #     y=y[10:]
            
        
        xaxis.append(x)
        x=np.insert(x,0,0)
        counts, bins = np.histogram(data, x)
        indexing = np.cumsum(counts)
        vals = []
        
        data = np.sort(data)
        for i in range(len(indexing)):
            if i == 0:
                vals.append(np.std(data[:indexing[i]+1]))

            if i > 0:
                vals.append(np.std(data[indexing[i-1]+1:indexing[i]+1]))
                
        total_stds.append(vals)   
            
        x=x[1:]    
         
            
        theo_p = np.array([theo_pdf(ne, k) for k in x])
        ax.set_xlabel(r'$k$', fontsize=21)
        ax.set_xscale('linear')
        ax.set_ylabel(r'$p(k)$', fontsize=21)
        ax.set_yscale('log')
        
        
        ax.plot(x, y, zorder=2.5, markersize = 6, marker='o', color=colors[mi], linestyle = 'None', markeredgewidth=0.4, markeredgecolor='black', label=mylegend[mi])
        x = np.linspace(x[0], max(x), num=1000, endpoint=True, dtype='float')
        theo_p = np.array([theo_pdf(ne, k) for k in x])
        
        idx = x >= ne+1
        x = x[idx]
        theo_p = theo_p[idx]
        
        ax.plot(x, theo_p, zorder=-1, color='black')
        plt.legend(fontsize = 'x-large', loc='upper right')
        mi+=1
    
    plt.show()
    print(total_stds)
    print(xaxis)
    



def gof_test():

    num_vertices = 200000
    num_edges = [4, 8, 16]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel('theo quantiles')
    ax1.set_ylabel('sample quantiles')

    ax2.set_xlabel('k')
    ax2.set_ylabel('prob')

    quantiles = rnd.uniform(0.0, 1.0, size=20)
    for ne in num_edges:
        print('num edges', ne)
        m0 = ne*(ne+1)
        g = build_graph_ra(m0, num_vertices, ne)
        dd = g.get_degree_dist()
        # qq plot
        prob = np.array(dd) / sum(dd)
        print(sum(prob))
        theo_q = [theo_ppf(ne, q) for q in quantiles]
        print(theo_q)
        empirical_q = [eppf(prob, q) for q in quantiles]
        print(empirical_q)
        ax1.plot(theo_q, empirical_q, '+')
        ax1.plot(theo_q, theo_q)
        # cum density plot
        k_s = np.linspace(0, 100, endpoint=True, num=100, dtype='int')
        theo_cums = [theo_cdf(ne, k) for k in k_s]
        emp_cums = [ecdf(prob, k) for k in k_s]
        ax2.plot(k_s, theo_cums)
        ax2.plot(k_s, emp_cums, '+')

    plt.show()


def ks_gof_test():
    # Kolmogorov-Smirnov gof test

    num_vertices = 100000
    num_edges = [2]
    
    

    for ne in num_edges:
        print('num edges', ne)
        m0 = 2*ne +1
        g = build_graph_ra(m0, num_vertices, ne)
        dd = g.get_degree_dist()
        degrees = expand_counts(dd)
        # generate some theoretical values
        urvs = rnd.uniform(0.0, 1.0, 1000)
        theo_degrees = np.array([theo_ppf(ne, u) for u in urvs])
        ks_res = ks_2samp(degrees, theo_degrees)
        print(ks_res)
        print('dcrit', 1.36*np.sqrt(1/len(theo_degrees)+1/len(degrees)))

def k1_calc():
    # estimate the largest degree versus N relationship
    num_vertices = 1000000
    ne = 2
    m0 = 5
    num_paths = 10

    rnd.seed(12345)
    k1_s = []
    sample_pts = np.array([i*i for i in range(int(np.sqrt(m0)+1), int(np.sqrt(m0+num_vertices)+1), 100)])
    tens = np.array([np.power(10, i) for i in np.arange(1, 10, 0.25)])
    tens = tens.astype('int')
    k1_10s = []
    

    for path in range(num_paths):
        print('path = ', path)
        k1 = []
        g = SimpleGraph()
        k1_10 = []
        xaxis = []
        edges = []
        for i in range(m0):
            g.add_vertex()

        edges = []
        for source in range(m0 - 1):
            g, edges = add_edge(g, source, ne, edges)

        for source in range(m0, m0 + num_vertices):
            g.add_vertex()
            g, edges = add_edge(g, source, ne, edges)
            if source in sample_pts:
                k1.append(g.get_max_vertex_degree())
            if source in tens:
                xaxis.append(source)
                k1_10.append(g.get_max_vertex_degree())
        
        k1_s.append(k1)
        k1_10s.append(k1_10)

        
    fig, ax = plt.subplots()
    # plot <k1> versus sqrt(N)
    
    k1_10s = list(zip(*k1_10s))
    k1_avg = np.array([np.mean(x) for x in k1_10s])
    k1_sd = np.array([np.std(x) for x in k1_10s])
    root_st = tens
    plt.xlabel(r'$N$', fontsize=21)
    plt.ylabel(r'$<k_1>$', fontsize=21)
    ax.set_xscale('log')
    ax.set_yscale('linear')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.plot(xaxis, k1_avg, linestyle='None', marker = 'o', color='#1f77b4', zorder=1.5)
    ax.errorbar(xaxis, k1_avg, yerr=k1_sd/np.sqrt(num_paths), linestyle='None', color='#1f77b4', zorder=1.5, capsize=4)

    
    
    logx = np.log(xaxis)
    logy = k1_avg
    
    
    p = np.polyfit(np.log(xaxis), k1_avg, 1)
    
    
    print(k1_avg)
    print(k1_sd/np.sqrt(num_paths))
    
    
    print(p[0],p[1])
    ax.plot(xaxis, p[0] * np.log(xaxis) + p[1], color='black', label=r'$Slope = 2.399 \pm 0.082 $')
    
    ax.plot(xaxis, 2.4663 * np.log(xaxis)+2, color='black', linestyle='dashed')
    # ax.plot(xaxis, 2.46*np.log(xaxis) + 2)
    # coeffs, pcov = np.polyfit(logx,logy,deg=1, cov=True)
    # poly = np.poly1d(coeffs)
    
    
    # print(poly)
    # yfit = lambda x: np.exp(poly(np.log(x)))
    # ax.plot(xaxis,yfit(xaxis), color = 'black',linewidth=1.7,  label=r'$Slope = 0.524 \pm 0.028 $', zorder=-2)
    
    plt.legend(loc='lower right', fontsize='x-large')

    plt.show()

    

    
    










def data_collapse():
    rnd.seed(12345)
    m0 = 10
    num_vertices = [100, 1000, 10000, 100000, 1000000]
    ne = 5

    for nv in num_vertices:
        print('data collapse, num vertices =', nv)
        g = build_graph_ra(m0, nv, ne)
        dd = g.get_degree_dist()
        prob = np.array(dd) / sum(dd)
        ks = np.array([k for k in range(len(dd))])
        idx = prob > 0.0
        prob = prob[idx]
        ks = ks[idx]
        idx = ks >= ne
        prob = prob[idx]
        ks = ks[idx]
        theo_prob = np.array([theo_pdf(ne, k) for k in ks])
        y = np.log10(prob / theo_prob)
        x = np.log10(ks/np.log10(nv))
        plt.plot(x, y, marker='o', markersize=4, linestyle='None')

    plt.xlabel('k/log(N)')
    plt.ylabel(r'$p_N(k)/p_\inf(k)$')
    plt.show()


if __name__ == '__main__':

    # degree_dist()
    #gof_test()
    # ks_gof_test()
    #data_collapse()
    k1_calc()
