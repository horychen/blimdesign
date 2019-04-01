from pylab import *
mpl.rcParams['legend.fontsize'] = 14
font = {'family' : 'Times New Roman', #'serif',
        'weight' : 'normal',
        'size'   : 14}
mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
mpl.rcParams['font.serif'] = ['Times New Roman']

d = {}
index_bias = 0
def load_random_designs(run_integer, d, bool_no_reference=False):
    global index_bias
    with open('../pop/run#%d/iemdc_data.txt'%(run_integer), 'r') as f:
        buf = f.readlines()
    for line in buf:
        l = [float(el) for el in line[:-1].split(',')]
        # if l[0] == 7:
        #     continue
        # print l
        # if l[0] == 21:
        #     continue
        if bool_no_reference == True:
            # print d[l[0]]
            # print l[1:6]
            # quit()
            d[l[0]+index_bias] = (d[l[0]][0], l[6:11], l[11:16], l[16:21])    
        else:
            d[l[0]+index_bias] = (l[1:6], l[6:11], l[11:16], l[16:21])
    index_bias += 1000


# run_integer = 94 # local test

# run_integer = 299
run_integer = 298
load_random_designs(run_integer, d)
# run_integer = 399
# load_random_designs(run_integer, d)

run_integer = 395 # coarse solving step
load_random_designs(run_integer, d, bool_no_reference=True)

run_integer = 295 # fine solving step
load_random_designs(run_integer, d, bool_no_reference=True)


for key, item in d.iteritems():
    print key, item
# quit()
# torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle
list_ylabel = ['Torque [Nm]', 'Torque ripple [%]', 'Force magnitude [N]', 'Force error magnitude [%]', 'Force error angle [deg]']
list_label = ['Tran. Ref.', 'Tran. w/ 2 Time Sect.', 'Rot. Eddy Current', 'Rot. Static']
list_color = ['k', 'tomato', 'limegreen', 'cornflowerblue'] # stackoverflow: named color in matplotlib

def data_as_list(fea_model, performance, d, index_range=[0,999]):
    # for key in [key for key, item in d.iteritems() if key>=index_range[0] and key<=index_range[1]]:
    #     print int(key)

    if performance == 1 or performance == 3: # [%]
        # print  [item[fea_model][performance]*100 for key, item in d.iteritems() if key>=index_range[0] and key<=index_range[1]]
        return [item[fea_model][performance]*100 for key, item in d.iteritems() if key>=index_range[0] and key<=index_range[1]]

    # print  [item[fea_model][performance] for key, item in d.iteritems() if key>=index_range[0] and key<=index_range[1]]
    return [item[fea_model][performance] for key, item in d.iteritems() if key>=index_range[0] and key<=index_range[1]]

fig, axes = subplots(5, 1, sharex=True, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
for i, ax in zip(range(5), axes):
    # torque_TranRef  = 
    # torque_Tran2TSS = 
    # torque_ECRotate = 
    # torque_FEMMStat = 

    # print data_as_list(1,i,d)[15]

    ax.plot( data_as_list(0,i,d), label=list_label[0], color=list_color[0], alpha=1.00 )
    ax.plot( data_as_list(1,i,d), label=list_label[1], color=list_color[1], alpha=1.00 )
    ax.plot( data_as_list(2,i,d), label=list_label[2], color=list_color[2], alpha=1.00 )
    ax.plot( data_as_list(3,i,d), label=list_label[3], color=list_color[3], alpha=1.00 )
    ax.legend()
    ax.grid()
    ax.set_ylabel(list_ylabel[i])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Number of Random Designs')


# print 'Normalized Error plot'
# fig, ax = subplots(1, 1, sharex=True, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
# for i in range(5):
#     # Normalized error
#     ref = array(data_as_list(0,i,d))
#     myones = ones(len(ref))
#     temp = [None]*4
#     if i == 4:
#         temp = list_label
#     ax.plot( (i-0.1)*myones, (array(data_as_list(1,i,d)) - ref) / ref, label=temp[1], marker='_', color=list_color[1], alpha=1.00 )
#     ax.plot( (i+0.0)*myones, (array(data_as_list(2,i,d)) - ref) / ref, label=temp[2], marker='_', color=list_color[2], alpha=1.00 )
#     ax.plot( (i+0.1)*myones, (array(data_as_list(3,i,d)) - ref) / ref, label=temp[3], marker='_', color=list_color[3], alpha=1.00 )
# ax.legend()
# xticks(range(5), list_ylabel)
# ax.set_ylabel('Normalized Performance Error with respect to Transient FEA Reference')


print 'Error plot'
fig, axes = subplots(1, 5, sharex=True, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=0.8, hspace=None)
list_ylabel = ['Torque difference [p.u.]', 'Torque ripple difference [%]', 'Force magnitude difference [p.u.]', 'Force error magnitude difference [%]', 'Force error angle difference [deg]']
for i, ax in enumerate(axes):
    print '------------------------',
    print 'performance =', i

    print 'index<1000'
    ref = array(data_as_list(0,i,d))
    if i == 0 or i == 2: 
        denominator = ref # difference in p.u.
    else:
        denominator = 1 # difference in original dimention
    myones = ones(len(ref))
    y = (array(data_as_list(1,i,d)) - ref)/denominator
    ax.plot( (-0.1)*myones, y,        label=list_label[1], marker='_', color=list_color[1], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))
    y = (array(data_as_list(2,i,d)) - ref)/denominator
    ax.plot( (+0.0)*myones, y,        label=list_label[2], marker='_', color=list_color[2], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))
    y = (array(data_as_list(3,i,d)) - ref)/denominator
    ax.plot( (+0.1)*myones, y,        label=list_label[3], marker='_', color=list_color[3], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))

    print '1000<=index<2000'
    ref = array(data_as_list(0,i,d, index_range=[1000,1999]))
    myones = ones(len(ref))
    y = (array(data_as_list(1,i,d, index_range=[1000,1999])) - ref)/denominator
    ax.plot( (-0.1-0.1/3.)*myones, y, label=list_label[1], marker='_', color=list_color[1], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))
    y = (array(data_as_list(2,i,d, index_range=[1000,1999])) - ref)/denominator
    ax.plot( (+0.0-0.1/3.)*myones, y, label=list_label[2], marker='_', color=list_color[2], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))
    y = (array(data_as_list(3,i,d, index_range=[1000,1999])) - ref)/denominator
    ax.plot( (+0.1-0.1/3.)*myones, y, label=list_label[3], marker='_', color=list_color[3], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))

    print '2000<=index<3000'
    ref = array(data_as_list(0,i,d, index_range=[2000,2999]))
    myones = ones(len(ref))
    y = (array(data_as_list(1,i,d, index_range=[2000,2999])) - ref)/denominator
    ax.plot( (-0.1+0.1/3.)*myones, y, label=list_label[1], marker='_', color=list_color[1], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))
    y = (array(data_as_list(2,i,d, index_range=[2000,2999])) - ref)/denominator
    ax.plot( (+0.0+0.1/3.)*myones, y, label=list_label[2], marker='_', color=list_color[2], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))
    y = (array(data_as_list(3,i,d, index_range=[2000,2999])) - ref)/denominator
    ax.plot( (+0.1+0.1/3.)*myones, y, label=list_label[3], marker='_', color=list_color[3], alpha=1.00 )
    print '\tmean(var)=$%g(%g)$'%(np.mean(y),np.var(y))
    # if i == 4:
    #     for ind, el in enumerate(data_as_list(3,i,d, index_range=[2000,2999])):
    #         print ind, el, ref[ind], el-ref[ind], 'This is not in original order.'
    #         quit()


    lower, upper = ax.get_ylim()
    ax.text(-0.1-0.1/3., lower + 0.15*(upper-lower), 'Coarse step',   rotation=90, color='tomato')
    ax.text(-0.1+0.0000, lower + 0.15*(upper-lower), 'Regular step',  rotation=90, color='tomato')
    ax.text(-0.1+0.1/3., lower + 0.15*(upper-lower), 'Fine step',     rotation=90, color='tomato')
    ax.text(+0.0-0.1/3., lower + 0.15*(upper-lower), 'Coarse step',   rotation=90, color='limegreen')
    ax.text(+0.0+0.0000, lower + 0.15*(upper-lower), 'Regular step',  rotation=90, color='limegreen')
    ax.text(+0.0+0.1/3., lower + 0.15*(upper-lower), 'Fine step',     rotation=90, color='limegreen')
    ax.text(+0.1-0.1/3., lower + 0.15*(upper-lower), 'Coarse step',   rotation=90, color='cornflowerblue')
    ax.text(+0.1+0.0000, lower + 0.15*(upper-lower), 'Regular step',  rotation=90, color='cornflowerblue')
    ax.text(+0.1+0.1/3., lower + 0.15*(upper-lower), 'Fine step',     rotation=90, color='cornflowerblue')

    ax.set_ylabel(list_ylabel[i])
    ax.grid()
    # Remove xticks
    # ax.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off
# ax.legend()
xticks([-0.1, 0, 0.1], list_label[1:])
# ax.set_ylabel('')

for ax in fig.axes:
    matplotlib.pyplot.sca(ax)
    plt.xticks(rotation=60)

fig.tight_layout()
fig.savefig(r'D:\OneDrive\[00]GetWorking\31 Bearingless_Induction_FEA_Model\p2019_iemdc_bearingless_induction full paper\images\random_design_plots.png', dpi=150)

show()


