import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(4, 13), sharey=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
lol=[[0.0, 0.0006325185340172684, 0.4715521189668933, 0.4866238907164036, 0.49245708834264457, 0.4949496996987989, 1.0983848711245205, 1.3911893847686787, 1.6926750665319807, 2.2697059191972486, 3.11121858556035, 3.6916481369096656, 4.845188070277649, 6.671777209416754, 8.468766010034319, 11.109583690203454, 14.504616255508349, 18.46858249322042, 25.06128871232086, 28.920153341807076, 40.756373189423456, 55.42677533307088, 79.79783483556709, 226.0], [0.000774553837598387, 0.4864466217634282, 0.49218054786274595, 0.49492492410653516, 1.0987616507648683, 1.3916938285997278, 1.6854858328680598, 2.2653531419187685, 3.0992336733302808, 3.6978896725817454, 4.803544440528696, 6.5313055669976405, 8.545746121101175, 11.159954447791597, 14.56585135629932, 18.4500076017766, 23.654981918850734, 31.048302998933348, 40.31692863992637, 50.195580259132576, 65.52036343528138, 94.91531313053055], [0.001000410752937932, 0.4949590203198941, 1.0986048444119842, 1.3927228107751923, 1.6886804732398546, 2.2617911443485204, 3.121598658712601, 3.7083355493183396, 4.842041611542533, 6.583824400932696, 8.539701575262782, 11.112862805974965, 14.08685939931696, 18.70257290555915, 24.145870098940357, 30.776478848987807, 41.68908150448825, 55.62246304394126, 67.58350085440173, 82.69184584364412, 99.82464572842746], [0.0, 1.107503158457936, 1.6871967179687395, 2.2601609920336183, 3.1184552173338624, 3.7003331442455787, 4.84325387545312, 6.560900648450208, 8.555265300144645, 11.112416708672175, 14.22856861324217, 18.626276886985778, 24.072631069887112, 30.738847719467834, 40.89792056531806, 52.32143072418081, 69.69268493927653, 91.92411331784577, 108.11668226562902, 112.99608406389181], [0.49949687365233986, 3.1187132129220547, 3.701091890977271, 4.838675096487325, 6.554636977319271, 8.546751449505418, 11.105068134782943, 14.272758794473553, 18.571144603159425, 23.978894790176657, 31.00704204071407, 40.9771998517165, 52.62723416281466, 68.59539729598649, 89.73918235657337, 118.72328017607101, 136.91328702235336, 105.58548910813494], [0.0, 3.148256415403849, 6.551769758348001, 8.548661576314624, 11.1366025047443, 14.24431636469625, 18.564443613979897, 23.917652735420237, 31.09920100969789, 40.650786865471204, 52.86919386364448, 68.80575065627897, 89.03325062673339, 115.13577977405551, 158.14789852243118, 178.38878273595506, 102.09710173074544]]
xol=[[   2.        ,    3.        ,    4.47213595,    6.4807407 ,
          8.48528137,   10.95445115,   14.4222051 ,   19.33907961,
         25.82634314,   33.76388603,   44.15880433,   57.57603668,
         74.89993324,   97.7036335 ,  127.43625858,  166.1144184 ,
        216.194357  ,  281.14053425,  365.42577906,  474.9705254 ,
        617.71190696,  803.60438028, 1045.05837158, 1358.82596384], [   4.47213595,    6.4807407 ,    8.48528137,   10.95445115,
         14.4222051 ,   19.33907961,   25.82634314,   33.76388603,
         44.15880433,   57.57603668,   74.89993324,   97.7036335 ,
        127.43625858,  166.1144184 ,  216.194357  ,  281.14053425,
        365.42577906,  474.9705254 ,  617.71190696,  803.60438028,
       1045.05837158, 1358.82596384], [   8.48528137,   10.95445115,   14.4222051 ,   19.33907961,
         25.82634314,   33.76388603,   44.15880433,   57.57603668,
         74.89993324,   97.7036335 ,  127.43625858,  166.1144184 ,
        216.194357  ,  281.14053425,  365.42577906,  474.9705254 ,
        617.71190696,  803.60438028, 1045.05837158, 1358.82596384,
       1766.83332547], [  14.4222051 ,   19.33907961,   25.82634314,   33.76388603,
         44.15880433,   57.57603668,   74.89993324,   97.7036335 ,
        127.43625858,  166.1144184 ,  216.194357  ,  281.14053425,
        365.42577906,  474.9705254 ,  617.71190696,  803.60438028,
       1045.05837158, 1358.82596384, 1766.83332547, 2296.79559387], [  33.76388603,   44.15880433,   57.57603668,   74.89993324,
         97.7036335 ,  127.43625858,  166.1144184 ,  216.194357  ,
        281.14053425,  365.42577906,  474.9705254 ,  617.71190696,
        803.60438028, 1045.05837158, 1358.82596384, 1766.83332547,
       2296.79559387, 2985.81245225], [  57.57603668,   74.89993324,   97.7036335 ,  127.43625858,
        166.1144184 ,  216.194357  ,  281.14053425,  365.42577906,
        474.9705254 ,  617.71190696,  803.60438028, 1045.05837158,
       1358.82596384, 1766.83332547, 2296.79559387, 2985.81245225,
       3882.07805177]]

y = []
for x in range(0,len(xol)):
    print(x)
    length = len(xol[x])
    yi = np.zeros(length)
    y.append(yi)
print(y)   

ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
plt.gcf().subplots_adjust(bottom=0.15)

ax1.plot( xol[0], y[0], marker=',', linestyle='None', color=colors[0])
ax1.errorbar( xol[0],y[0],yerr=lol[0], linestyle='None',color=colors[0], capsize =3, elinewidth=2)
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(11)
ax2.plot( xol[1], y[1], marker=',', linestyle='None', color=colors[1])
ax2.errorbar( xol[1],y[1],yerr=lol[1], linestyle='None', color=colors[1], capsize =3, elinewidth=2)
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
	label.set_fontsize(11)
ax3.plot( xol[2], y[2], marker=',', linestyle='None', color=colors[2])
ax3.errorbar( xol[2],y[2],yerr=lol[2], linestyle='None', color=colors[2], capsize =3, elinewidth=2)
for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
	label.set_fontsize(11)
ax4.plot( xol[3], y[3], marker=',', linestyle='None', color=colors[3])
ax4.errorbar( xol[3],y[3],yerr=lol[3], linestyle='None', color=colors[3], capsize =3, elinewidth=2)
for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
	label.set_fontsize(11)
ax5.plot( xol[4], y[4], marker=',', linestyle='None', color=colors[4])
ax5.errorbar( xol[4],y[4],yerr=lol[4], linestyle='None', color=colors[4], capsize =3, elinewidth=2)
for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
	label.set_fontsize(11)
ax6.plot( xol[5], y[5], marker=',', linestyle='None', color=colors[5])
ax6.errorbar( xol[5],y[5],yerr=lol[5], linestyle='None', color=colors[5], capsize =3, elinewidth=2)
for label in (ax6.get_xticklabels() + ax6.get_yticklabels()):
	label.set_fontsize(11)
plt.tight_layout()
fig.text(0.5, 0.01, r'$k$', ha='center', fontsize=15)
fig.text(0.01, 0.5, r'$Standard \quad Error$', va='center', rotation='vertical', fontsize=14)


#%%

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(4, 13))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
lol=[[0.0, 0.18246560765962694, 0.001224596704152585, 0.48977127019464306, 0.48991711702403284, 0.4899728295729848, 0.4907564811576711, 1.0382428439297777, 1.27273096043078, 1.4653142701989743, 1.5187859543224844], [0.0, 0.010488139678000398, 0.49690200902263476, 0.49694117559818446, 0.4967307190302321, 1.0958464402718269, 1.3697825280048685, 1.6313623176265146, 2.1318027145462333, 2.7616843779705254, 2.6284395994374985], [0.0, 0.02173013020506225, 0.4991433016730698, 1.111601274065952, 1.4007117521306862, 1.6853979251827496, 2.2396450964534393, 3.032286436934858, 3.5244762014085897, 4.378831864177172, 4.547441164774002], [0.0, 1.1176681384290368, 1.7032138537402115, 2.277149150688317, 3.1297941165304195, 3.6837012493507637, 4.768718380389794, 6.320963601660835, 8.003423758716172, 8.681752218324382], [0.505448113196499, 3.153737709188118, 3.727491437109114, 4.868667841645682, 6.549092778450979, 8.454576673673913, 10.865531507857492, 13.678151301010352, 17.230727402987878], [0.0, 3.165857238308626, 6.6109318791939025, 8.60843270431771, 11.156540853739141, 14.188927508680695, 18.288148774263295, 23.20681391771986, 28.88907316400822, 15.697277266424942]]
xol=[[ 1.        ,  2.        ,  3.        ,  4.47213595,  6.4807407 ,
        8.48528137, 10.95445115, 14.4222051 , 19.33907961, 25.82634314,
       33.76388603],[ 3.        ,  4.47213595,  6.4807407 ,  8.48528137, 10.95445115,
       14.4222051 , 19.33907961, 25.82634314, 33.76388603, 44.15880433,
       57.57603668],[ 6.4807407 ,  8.48528137, 10.95445115, 14.4222051 , 19.33907961,
       25.82634314, 33.76388603, 44.15880433, 57.57603668, 74.89993324,
       97.7036335 ], [ 14.4222051 ,  19.33907961,  25.82634314,  33.76388603,
        44.15880433,  57.57603668,  74.89993324,  97.7036335 ,
       127.43625858, 166.1144184 ],[ 33.76388603,  44.15880433,  57.57603668,  74.89993324,
        97.7036335 , 127.43625858, 166.1144184 , 216.194357  ,
       281.14053425], [ 57.57603668,  74.89993324,  97.7036335 , 127.43625858,
       166.1144184 , 216.194357  , 281.14053425, 365.42577906,
       474.9705254 , 617.71190696]]
                       
# for x in range(0,len(lol)):
#     lol[x]=lol[x][1:]

# for y in range(0,len(xol)):
#     xol[y]=xol[y][1:]                                                     
                                                     
                                                     
                                                     
y = []
for x in range(0,len(xol)):

    length = len(xol[x])
    print(length)
    yi = np.zeros(length)
    y.append(yi)


ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
# from matplotlib.ticker import StrMethodFormatter, NullFormatter

# ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax1.yaxis.set_minor_formatter(NullFormatter())
# ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax1.xaxis.set_minor_formatter(NullFormatter())
# ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax2.yaxis.set_minor_formatter(NullFormatter())
# ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax2.xaxis.set_minor_formatter(NullFormatter())
# ax3.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax3.yaxis.set_minor_formatter(NullFormatter())
# ax3.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax3.xaxis.set_minor_formatter(NullFormatter())
# ax4.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax4.yaxis.set_minor_formatter(NullFormatter())
# ax4.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax4.xaxis.set_minor_formatter(NullFormatter())
# ax5.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax5.yaxis.set_minor_formatter(NullFormatter())
# ax5.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax5.xaxis.set_minor_formatter(NullFormatter())
# ax6.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax6.yaxis.set_minor_formatter(NullFormatter())
# ax6.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
# ax6.xaxis.set_minor_formatter(NullFormatter())







plt.gcf().subplots_adjust(bottom=0.15)

ax1.plot( xol[0], y[0], marker=',', linestyle='None', color=colors[0])
ax1.errorbar( xol[0],y[0],yerr=lol[0], linestyle='None',color=colors[0], capsize =3, elinewidth=2)
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(11)
ax2.plot( xol[1], y[1], marker=',', linestyle='None', color=colors[1])
ax2.errorbar( xol[1],y[1],yerr=lol[1], linestyle='None', color=colors[1], capsize =3, elinewidth=2)
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
	label.set_fontsize(11)
ax3.plot( xol[2], y[2], marker=',', linestyle='None', color=colors[2])
ax3.errorbar( xol[2],y[2],yerr=lol[2], linestyle='None', color=colors[2], capsize =3, elinewidth=2)
for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
	label.set_fontsize(11)
ax4.plot( xol[3], y[3], marker=',', linestyle='None', color=colors[3])
ax4.errorbar( xol[3],y[3],yerr=lol[3], linestyle='None', color=colors[3], capsize =3, elinewidth=2)
for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
	label.set_fontsize(11)
ax5.plot( xol[4], y[4], marker=',', linestyle='None', color=colors[4])
ax5.errorbar( xol[4],y[4],yerr=lol[4], linestyle='None', color=colors[4], capsize =3, elinewidth=2)
for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
	label.set_fontsize(11)
ax6.plot( xol[5], y[5], marker=',', linestyle='None', color=colors[5])
ax6.errorbar( xol[5],y[5],yerr=lol[5], linestyle='None', color=colors[5], capsize =3, elinewidth=2)
for label in (ax6.get_xticklabels() + ax6.get_yticklabels()):
	label.set_fontsize(11)
plt.tight_layout()
fig.text(0.5, 0.01, r'$k$', ha='center', fontsize=15)
fig.text(0.01, 0.5, r'$Standard \quad Error$', va='center', rotation='vertical', fontsize=14)

#%%
fig, ax = plt.subplots(figsize=(7,2))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

std=[0.0, 0.3, 0.0012467209556462746, 0.001766092717973038, 0.002367170585676313, 0.003052215254610017, 0.003822617680270503, 0.4911866615771784, 0.49392033505640753, 0.49561463009622003, 0.8080664564309388, 1.1086145268171146, 1.118919530203281, 1.101484517990653, 1.6882122722389914, 2.2784553165198735, 2.2534686154592842, 2.8416211287250537, 3.534961023804385, 4.404625592125316, 4.294746126450019, 5.102372555321358, 6.786714109181412, 7.692971999287283, 13.62666321640126, 8.499411744350311, 29.090204536922734]
array = [  1.        ,   2.        ,   3.        ,   4.        ,
         5.        ,   6.        ,   7.        ,   8.48528137,
        10.48808848,  12.9614814 ,  15.96871942,  19.4422221 ,
        23.4520788 ,  27.92848009,  33.86738844,  41.35214626,
        49.83974318,  59.79130372,  71.74956446,  86.17424209,
       103.60984509, 124.51505933, 149.43560486, 179.28747865,
       215.62003617, 258.98069426, 373.45548597]

                                                                                                    
y=np.zeros(len(array))
ax.set_xlabel(r'$k$', fontsize=21)
ax.set_ylabel(r'$\sigma$', fontsize=21)

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()

ax.set_xscale('log')
ax.plot( array, y, marker=',', linestyle='None', color=colors[0])
ax.errorbar( array,y,yerr=std, linestyle='None',color=colors[0], capsize =3, elinewidth=2)
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(11)

#%%
fig, ax = plt.subplots(figsize=(7,2))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

std=[0.0, 0.24206145913796356, 0.0013331070205935378, 0.0017395198151149227, 0.0022240241371352665, 0.0027965382384027545, 0.0034660639625922465, 0.4905485886384291, 0.49190043414792456, 0.4938147751974863, 0.800314118118741, 1.0945505650884864, 1.1053125060045366, 1.0909563631474077, 1.6713392079371565, 2.1948590452176426, 2.2698840947756604, 2.867735225826085, 3.5215112759533604, 5.109733358992424, 0.0]
array=[  1.        ,   2.        ,   3.        ,   4.        ,
         5.        ,   6.        ,   7.        ,   8.48528137,
        10.48808848,  12.9614814 ,  15.96871942,  19.4422221 ,
        23.4520788 ,  27.92848009,  33.86738844,  41.35214626,
        49.83974318,  59.79130372,  71.74956446,  86.17424209,
       103.60984509]

                                                                                                    
y=np.zeros(len(array))
ax.set_xlabel(r'$k$', fontsize=21)
ax.set_ylabel(r'$\sigma$', fontsize=21)

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()

ax.set_xscale('log')
ax.plot( array, y, marker=',', linestyle='None', color=colors[2])
ax.errorbar( array,y,yerr=std, linestyle='None',color=colors[2], capsize =3, elinewidth=2)
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(11)
