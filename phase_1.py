"""
main entry for networks project
author - Talia Rahall, March 2021
"""
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from scipy.stats import linregress, ks_2samp

from log_bin import logbin
from simple_graph import SimpleGraph


def expand_counts(counts: [], zeros: bool = True) -> np.array:
    # expand counts of k to an array
    # counts - list of counts
    # zeros -  True to remove zeros, false to keep
    data = np.array([k for k, count in enumerate(counts) for i in range(count)])
    return data[data > 0] if zeros else data


def theo_pdf(m: int, k: int) -> float:
    return 0.0 if k < m else 2.0 * float(m * (m+1)) / float((k * (k+1) * (k+2)))


def theo_ppf(m: int, q: float) -> int:
    if q == 1.0:
        q -= 1e-8
    c = 2.0 - m*(m+1)/(1.0-q)
    k = (-3 + np.sqrt(9.0 - 4*c))*0.5
    return ceil(k)


def theo_cdf(m: int, k: int) -> float:
    return 0.0 if k < m else 1.0 - float(m*(m+1))/float((k+1)*(k+2))


def ecdf(prob: np.array, k: int) -> float:
    if k < 0:
        return 0.0
    elif k >= len(prob):
        return 1.0
    else:
        # s = 0.0
        # for i in range(k+1):
        #     s += prob[i]
        # return s
        return np.sum(prob[:k+1])


def eppf(prob: np.array, q: float) -> int:
    # empirical percentage point function
    s = 0.0
    k = 0
    while s < q:
        if k == len(prob):
            break
        s += prob[k]
        k += 1
    return k - 1


def gof_test_pa():

    num_vertices = 200000
    num_edges = [2,4,8, 16, 32, 64]
    plt.clf()
    
    ylabel = [r'$Sample \, Quantiles$', '', '', r'$Sample \, Quantiles$', '', '']
    xlabal = ['','','', r'$Theoretical \, Quantiles$', r'$Theoretical \, Quantiles$', r'$Theoretical \, Quantiles$']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    mylegend=[r'$m=2$', r'$m=4$', r'$m=8$', r'$m=16$', r'$m=32$', r'$m=64$']
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    i=0
    quantiles = rnd.uniform(0.0, 1.0, size=50)
    fig = plt.figure(figsize=(20,20))
    for ne in num_edges:
        print('num edges', ne)
        m0 = ne*(ne+1)
        g = build_graph_pa(m0, num_vertices, ne)
        dd = g.get_degree_dist()
        # qq plot
        prob = np.array(dd) / sum(dd)
      
        theo_q = [theo_ppf(ne, q) for q in quantiles]
        
    
        empirical_q = [eppf(prob, q) for q in quantiles]
   
             
        ax = plt.subplot(2,3,i+1)
        ax.plot(theo_q, empirical_q, zorder=2.5, markersize = 6, marker='o', color=colors[i], linestyle = 'None', markeredgewidth=0.4, markeredgecolor='black', label=mylegend[i])
        ax.plot(theo_q, theo_q, zorder=-1, color='black')
        plt.legend(loc='lower right', fontsize='x-large')
        ax.text(0.05, 0.95, labels[i], transform=ax.transAxes,fontsize=16, fontweight='bold', va='top') 

        plt.xlabel(xlabal[i], fontsize=14)
        plt.ylabel(ylabel[i], fontsize=14)

        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

            
        plt.legend(loc='lower right', fontsize='x-large')
            
        i+=1
    plt.show()


def ks_gof_test():
    # goodness of fit test

    from matplotlib.ticker import StrMethodFormatter, NullFormatter
    plt.clf()
    num_vertices = 100000
    num_edges = [2, 4, 8, 16, 32, 64]
    KS = []
    P = []
    i=0
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    mylegend = [r'$m=2$', r'$m=4$', r'$m=8$', r'$m=16$', r'$m=32$', r'$m=64$']
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    ylabel = [r'$F(k)$', '', '', r'$F(k)$', '', '']
    xlabal = ['','','', r'$k$', r'$k$', r'$k$']
    numbers = [40, 60, 80, 130, 250, 350]
    xlims = [1.75, 3.59, 7.06, 13.7, 25.5, 55.1]
    xlims2 = [44, 64, 88, 141, 280, 390]
    num2 = [15, 19, 23, 46, 79, 142]
    
    num3 = [2, 3, 3, 4, 6, 8]
    
    for ne in num_edges:
        data = []
        print('num edges', ne)
        m0 = ne*(ne+1)
        g = build_graph_pa(m0, num_vertices, ne)
        dd = g.get_degree_dist()
        prob = np.array(dd) / sum(dd)
        degrees = expand_counts(dd)
        data.extend(degrees)
   
        # generate some theoretical values
        urvs = rnd.uniform(0.0, 1.0, 1000)
        theo_degrees = np.array([theo_ppf(ne, u) for u in urvs])
        ks_res = ks_2samp(degrees, theo_degrees)
        
        print(ks_res)
        KS.append(ks_res[0])
        P.append(ks_res[1])
        

        k_s = np.linspace(ne, numbers[i], endpoint=True, num=10000)
        theo_cums = [theo_cdf(ne, k) for k in k_s]     
        ax = plt.subplot(2,3,i+1)
        ax.loglog(k_s, theo_cums, linestyle= '-', color='black')
        
        k_s = np.linspace(ne, numbers[i], endpoint=True, num=(numbers[i]-ne), dtype='int')
        # k_s2 = np.linspace(num2[i]+num3[i], numbers[i], endpoint=True, num=10, dtype='int')
        # k_s = np.concatenate((k_s,k_s2))
        
        emp_cums = [ecdf(prob, k) for k in k_s]
       
        #formatting
        print('dcrit', 1.36*np.sqrt(1/len(theo_degrees)+1/len(degrees)))
        ax.loglog(k_s, emp_cums, marker = 'o', color = colors[i], markersize= 6,markeredgewidth=0.4, markeredgecolor='black', linestyle='None', label=mylegend[i])
        plt.legend(loc='lower right', fontsize='x-large')
        ax.text(0.05, 0.95, labels[i], transform=ax.transAxes,fontsize=16, fontweight='bold', va='top') 
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        plt.xlabel(xlabal[i], fontsize=21)
        plt.ylabel(ylabel[i], fontsize=21)
        plt.ylim(min(theo_cums), 1.01)
        plt.xlim(xlims[i], numbers[i])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        i+=1
        


    plt.show()

    
    

def draw_graph(g: SimpleGraph, pause: float = 0.05):
    n = g.get_num_vertices()
    v_pos = np.zeros((n, 2), dtype='float')

    radius = 0.9
    for i in range(n):
        u = rnd.uniform()
        v_pos[i, 0] = radius * np.cos(2.0 * u * np.pi)
        v_pos[i, 1] = radius * np.sin(2.0 * u * np.pi)
        plt.annotate(str(i) + ',' + str(len(g.get_neighbours(i))), (v_pos[i, 0], v_pos[i, 1]))

    for i in range(n):
        n = g.get_neighbours(i)
        for j in n:
            x_ = [v_pos[i, 0], v_pos[j, 0]]
            y_ = [v_pos[i, 1], v_pos[j, 1]]
            plt.plot(x_, y_)
            plt.pause(pause)
    plt.show()


def degree_dist():
    # finds a numerical value for the degree distribution
    rnd.seed(12345)

    num_vertices = 100000
    num_edges = [2,4, 8, 16, 32, 64]
    m0 = [num*2 + 1 for num in num_edges]
    num_paths = 50
    y_data = []

    fig,  ax2 = plt.subplots()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    mylegend = [r'$m=2$', r'$m=4$', r'$m=8$', r'$m=16$', r'$m=32$', r'$m=64$']

    mi = 0
    total_stds = []
    xaxis = []
    for ne in num_edges:
        data = []
        

        for path in range(num_paths): 
            
            print('num edges', ne, 'path=', path)
            g = build_graph_pa(m0[mi], num_vertices, ne)
            
            dd = g.get_degree_dist()
            data.extend(expand_counts(dd))
        
        x, y = logbin(data, scale=1.3)
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
        
        

        x, y = np.log10(x[1:]), np.log10(y)
        ax2.set_xlabel(r'$k$', fontsize = 21)
        ax2.set_ylabel(r'$p_N(k)$', fontsize = 21)
        

        ax2.loglog(10**x, 10**y, marker = 'o', color = colors[mi], markersize= 6,markeredgewidth=0.4, markeredgecolor='black', linestyle='None', label=mylegend[mi], zorder=2)
        
        x1 = np.arange(ne, max(10**x)+5, 0.01)
        theo_prob = np.array([theo_pdf(ne, k) for k in x1])
        
        ax2.loglog(x1, theo_prob, color='black', zorder=-1)
       
        plt.legend(fontsize = 'x-large',  loc='lower left')
        plt.tight_layout()
        mi+=1
        plt.show()
   
    print(total_stds)
    print(xaxis)
    
def data_collapse():
    rnd.seed(12345)
    from matplotlib.ticker import StrMethodFormatter, NullFormatter
    m0 = 5
    num_vertices = [1000, 10000, 100000, 1000000]
    ne = 2
    num_paths = 100
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    i=0
    fig, ax = plt.subplots()
    for nv in num_vertices:
        data = []
        for path in range(num_paths):
            print('num vertices =', nv,'path =', path)
            g = build_graph_pa(m0, nv, ne)
            dd = g.get_degree_dist()
            data.extend(expand_counts(dd))
        x, prob = logbin(data, 1.3)
        idx = x >= ne
        x = x[idx]
        prob = prob[idx]
        theo_prob = np.array([theo_pdf(ne, k) for k in x])
        y = prob/theo_prob
        
        plt.loglog(x/np.sqrt(nv), y, marker = 'o', color = colors[i], markersize= 6,markeredgewidth=0.4, markeredgecolor='black', linestyle='None', label=r'$N=%i$'%(nv))
        plt.xticks(fontsize=16)
        plt.legend(loc='lower left', fontsize='x-large')
        plt.yticks(fontsize=16)
        plt.xlabel(r'$k \, / \, \sqrt{N}$', fontsize = 21)
        plt.ylabel(r'$p_E(k) \, / \,  p_\infty(k)$', fontsize = 21)

        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        i+=1

    plt.show()


def add_edge(g: SimpleGraph, source: int, ne: int, edges: []) -> ():
    while len(g.get_neighbours(source)) != ne:
        edge = edges[rnd.randint(0, len(edges))]
        target = edge[rnd.randint(0, 2)]
        if source != target and target not in g.get_neighbours(source):
            g.add_edge_slowly(source, target)
            edges.append((source, target))
    return g, edges


def build_graph_pa(m0: int, m: int, ne: int) -> SimpleGraph:
    """
        builds a graph based on random distributions.
        the graph is initialised by attempting to add ne edges to the initial m0 vertices
        following this additional vertices are added and connected to existing vertices
        this is done by keeping a list of existing edges and sampling this list uniformly.
        once an edge is selected then the end of the edge is selected again by randomly selecting
        the end.
        author - Talia Rahall, March 2021
    """
    # m0 - initial number of vertices
    # m - additional number of vertices
    # ne - number of edges per vertex

    if ne > m0:
        raise Exception('m0 must be greater than ne')

    print('**** preferential attachment - edges method ****')
    g = SimpleGraph()
    edges = []
    
    for i in range(m0):
        g.add_vertex()

    for source in range(m0-1):
       
        for iter_ in range(ne):
            target = rnd.randint(0, m0)
            
            if source != target and target not in g.get_neighbours(source):
                g.add_edge_slowly(source, target)
                edges.append((source, target))
                print(source+1, 'source O')
                print(len(edges), 'e')

    



    for source in range(m0, m0+m):
        g.add_vertex()
        
        g, edges = add_edge(g, source, ne, edges)

        
        if source in (54, 104, 154, 204, 254, 304, 354, 404, 454, 504, 554, 604):
        
            print(source+1, 'source V')
            print(len(edges), 'e')
            print(len(edges)/(source+1))   
        
       
    
    return g


def k1_calc():
    # estimate the largest degree versus N relationship
    
    ne = 2
    m0 = 5
    num_paths = 10
    num_vertices = 1000000-m0

    rnd.seed(12345)
    k1_s = []
    sample_pts = np.array([i*i for i in range(int(np.sqrt(m0)+1), int(np.sqrt(m0+num_vertices)+1), 100)])

    
    tens = np.array([np.power(10, i) for i in np.arange(1, 10, 0.25)])
    tens = tens.astype('int')
    k1_10s = []
    
    for path in range(num_paths):
        
        print('path = ', path)
        g = SimpleGraph()
        k1 = []
        k1_10 = []
        xaxis = []
        edges = []
        for i in range(m0):
            g.add_vertex()

        for source in range(m0 - 1):
            for iter_ in range(ne):
                target = rnd.randint(0, m0)
                if source != target and target not in g.get_neighbours(source):
                    g.add_edge_slowly(source, target)
                    edges.append((source, target))

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



        
    ax = plt.subplot(1, 2, 1)
    # plot k1s versus sqrt(N)
    
    plt.xlabel(r'$N$', fontsize=21)
    plt.ylabel(r'$k_1$', fontsize=21)
    root_st = np.sqrt(sample_pts)
    for k1_ in k1_s:
        ax.loglog(root_st**2, k1_)
    k1_s = list(zip(*k1_s))
    k1_avg = np.array([np.mean(x) for x in k1_s])
    k1_sd = np.array([np.std(x) for x in k1_s])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    res = linregress(np.log10(root_st**2), np.log10(k1_avg))

    #plt.loglog(root_st**2, k1_avg, color='black', label=r'$k_1 = {%.3f}N + {%.3f} $' % (float(res.slope), float(res.intercept)), linewidth=3.0)
    #plt.legend(loc='lower right', fontsize='x-large')
    ax.text(0.05, 0.95, 'A', transform=ax.transAxes,fontsize=16, fontweight='bold', va='top') 
    plt.show()

        
    

    ax = plt.subplot(1, 2, 2)
    # plot <k1> versus sqrt(N)
    root_st = np.sqrt(xaxis)
    k1_10s = list(zip(*k1_10s))
    k1_avg = np.array([np.mean(x) for x in k1_10s])
    
    k1_sd = np.array([np.std(x) for x in k1_10s])

    plt.xlabel(r'$N$', fontsize=21)
    plt.ylabel(r'$<k_1>$', fontsize=21)
    ax.loglog(root_st**2, k1_avg, '+', linestyle='None', marker = 'o', color='#1f77b4', zorder=1.5)
    ax.errorbar(root_st**2, k1_avg, yerr=k1_sd/np.sqrt(num_paths), linestyle='None', color='#1f77b4', zorder=1.5, capsize=4)
    print(k1_avg)
    print(k1_sd/np.sqrt(num_paths))
    logx = np.log(root_st**2)
    logy = np.log(k1_avg)
    coeffs, pcov = np.polyfit(logx,logy,deg=1, cov=True)
    poly = np.poly1d(coeffs)
    print(poly)
    yfit = lambda x: np.exp(poly(np.log(x)))
    ax.loglog(root_st**2,yfit(root_st**2), color = 'black',linewidth=1.7,  label=r'$Slope = 0.524 \pm 0.028 $', zorder=-2)
    plt.xticks(fontsize=16)
    plt.legend(loc='lower right', fontsize='x-large')
    plt.yticks(fontsize=16)
    ax.text(0.05, 0.95, 'B', transform=ax.transAxes,fontsize=16, fontweight='bold', va='top') 
    plt.show()
    print(res.slope, res.intercept)

    

    
    


    
    

    # # plot std of k1 versus sqrt(N)
    # plt.title(r'$\sigma$ versus $\sqrt{N}$')
    # plt.xlabel(r'$\sqrt{N}$')
    # plt.ylabel(r'$\sigma$')
    # plt.plot(root_st, k1_sd, '+')
    # res = linregress(np.sqrt(sample_pts), k1_sd)
    # print(res.slope, res.intercept)
    # predict = res.slope * root_st + res.intercept
    # plt.plot(root_st, predict)
    # plt.show()


if __name__ == '__main__':
    # degree_dist()
     # gof_test_pa()
    # ks_gof_test()
    # data_collapse()
    # k1_calc()
    build_graph_pa(5, 600, 4)
