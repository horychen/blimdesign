from pylab import *

d = {}

run_integer = 94
run_integer = 299
with open('../pop/run#%d/iemdc_data.txt'%(run_integer), 'r') as f:
    buf = f.readlines()
for line in buf:
    l = [float(el) for el in line[:-1].split(',')]
    d[l[0]+100] = (l[1:6], l[6:11], l[11:16], l[16:21])

# run_integer = 399
# with open('../pop/run#%d/iemdc_data.txt'%(run_integer), 'r') as f:
#     buf = f.readlines()
# for line in buf:
#     l = [float(el) for el in line[:-1].split(',')]
#     d[l[0]+200] = (l[1:6], l[6:11], l[11:16], l[16:21])

for key, item in d.iteritems():
    print key, item

# torque_average, normalized_torque_ripple, ss_avg_force_magnitude, normalized_force_error_magnitude, force_error_angle
list_ylabel = ['Torque [Nm]', 'Torque Ripple [%]', 'Force Mag. [N]', 'Force Mag. Err. [%]', 'Force Angle Err [deg]']
list_label = ['Tran. Ref.', 'Tran w/ 2 Sect.', 'Rot. Eddy Current', 'Rot. Static']
list_color = ['k', 'tomato', 'limegreen', 'cornflowerblue']

def data_as_list(fea_model, performance, d):
    # print [item[fea_model][performance] for key, item in d.iteritems()]
    return [item[fea_model][performance] for key, item in d.iteritems()]

fig, axes = subplots(5, 1, sharex=True, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
for i, ax in zip(range(5), axes):
    # torque_TranRef  = 
    # torque_Tran2TSS = 
    # torque_ECRotate = 
    # torque_FEMMStat = 

    print data_as_list(1,i,d)[15]

    ax.plot( data_as_list(0,i,d), label=list_label[0], color=list_color[0], alpha=1.00 )
    ax.plot( data_as_list(1,i,d), label=list_label[1], color=list_color[1], alpha=1.00 )
    ax.plot( data_as_list(2,i,d), label=list_label[2], color=list_color[2], alpha=1.00 )
    ax.plot( data_as_list(3,i,d), label=list_label[3], color=list_color[3], alpha=1.00 )
    ax.legend()
    ax.grid()
    ax.set_ylabel(list_ylabel[i])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Number of Random Designs')


print 'Normalized Error plot'
fig, ax = subplots(1, 1, sharex=True, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
for i in range(5):
    # Normalized error
    ref = array(data_as_list(0,i,d))
    myones = ones(len(ref))
    temp = [None]*4
    if i == 4:
        temp = list_label
    ax.plot( (i-0.1)*myones, (ref - array(data_as_list(1,i,d))) / ref, label=temp[1], marker='_', color=list_color[1], alpha=1.00 )
    ax.plot( (i+0.0)*myones, (ref - array(data_as_list(2,i,d))) / ref, label=temp[2], marker='_', color=list_color[2], alpha=1.00 )
    ax.plot( (i+0.1)*myones, (ref - array(data_as_list(3,i,d))) / ref, label=temp[3], marker='_', color=list_color[3], alpha=1.00 )
ax.legend()
xticks(range(5), list_ylabel)
ax.set_ylabel('Normalized Performance Error with respect to Transient FEA Reference')

print 'Error plot'
fig, axes = subplots(1, 5, sharex=True, dpi=150, figsize=(12, 6), facecolor='w', edgecolor='k')
for i, ax in enumerate(axes):
    ref = array(data_as_list(0,i,d))
    myones = ones(len(ref))
    ax.plot( (-0.1)*myones, (ref - array(data_as_list(1,i,d))) / ref, label=temp[1], marker='_', color=list_color[1], alpha=1.00 )
    ax.plot( (+0.0)*myones, (ref - array(data_as_list(2,i,d))) / ref, label=temp[2], marker='_', color=list_color[2], alpha=1.00 )
    ax.plot( (+0.1)*myones, (ref - array(data_as_list(3,i,d))) / ref, label=temp[3], marker='_', color=list_color[3], alpha=1.00 )
    ax.set_ylabel(list_ylabel[i])
ax.legend()
# xticks(range(5), list_ylabel)
# ax.set_ylabel('')


show()
