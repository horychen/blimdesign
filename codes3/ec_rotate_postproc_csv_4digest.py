# coding:u8
# execfile(r'D:\Users\horyc\OneDrive - UW-Madison\ec_rotate_postproc.py')
from pylab import *
# plt.style.use('ggplot')
from csv import reader as csv_reader
# mpl.rcParams['legend.fontsize'] = 14
# font = {'family' : 'Times New Roman', #'serif',
#         'weight' : 'normal',
#         'size'   : 14}
mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
mpl.rcParams['font.serif'] = ['Times New Roman']

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
def locallyZoomIn(ax, data_zoomed, ylim=None, loc1=None, loc2=None, y_axis_shrink=None, **kwarg):

    if(y_axis_shrink!=None):
        temp_tuple = (ylim[0]*y_axis_shrink, ylim[1]*y_axis_shrink)
        ylim = temp_tuple
        for ind, el in enumerate(data_zoomed):
            temp_tuple = (el[0], el[1]*y_axis_shrink)
            data_zoomed[ind] = temp_tuple

    axins = zoomed_inset_axes(ax, **kwarg) # zoom = 
    # axins.plot(Z2, extent=extent, interpolation="nearest", origin="lower")
    for ind, el in enumerate(data_zoomed):
        if ind==1:
            if len(data_zoomed)>1:
                axins.plot(el[0],el[1], 'r-', lw=0.75, alpha=0.5) # '--'
            else:
                axins.plot(el[0],el[1], 'k--', lw=0.75, alpha=0.5)
        else:
            # axins.plot(*el, lw=2)
            axins.plot(el[0],el[1], 'k--', lw=0.75, alpha=0.5)

    # sub region of the original image
    # x1, x2, y1, y2 = 1.2*pi, 1.25*pi, -0.78, -0.54
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)

    if ylim!=None:
        axins.set_ylim(ylim)

    plt.xticks(arange(  axins.viewLim.intervalx[0],
                        axins.viewLim.intervalx[1],
                        axins.viewLim.width/2), visible=False)
    plt.yticks(arange(  axins.viewLim.intervaly[0],
                        axins.viewLim.intervaly[1],
                        axins.viewLim.height/4), visible=False)

    if(y_axis_shrink==None):
        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        if loc1!=None and loc2!=None:
            # mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.5")
            mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="b", lw=1.5)
            
            # axins contains the zoomed-in box
            for axis in ['top','bottom','left','right']:
                axins.spines[axis].set_linewidth(1.5)
                axins.spines[axis].set_color('b')
        else:
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
        plt.draw()
    return axins


def whole_row_reader(reader):
    for row in reader:
        yield row[:]

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

f_name1 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 0Hz-SectioinAirGapBradial_atRotorSurface.csv'
f_name2 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 50Hz-SectioinAirGapBradial_atRotorSurface.csv'
f_name3 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 500Hz-SectioinAirGapBradial_atRotorSurface.csv'

# fig, ax = subplots(nrows=3, ncols=2, sharex=False, dpi=100, figsize=(16, 9), facecolor='w', edgecolor='k')


# import matplotlib.gridspec as gridspec
# gridspec.GridSpec(3,2, hspace=0.65, wspace=0.15)

# Plot figure with subplots of different sizes
fig = plt.figure(9, dpi=150, figsize=(16, 8), facecolor='w', edgecolor='k')
# set up subplot grid
plt.GridSpec(3,2, hspace=0.36, wspace=0.15) # hspace=0.65

# print ax
ind = 0
for f_name, freq in zip([f_name1, f_name2, f_name3], [7.5, 50, 500]):    
    angle_list = []
    Bairgap_list = []
    with open(f_name, 'r') as f:
        read_iterator = csv_reader(f, skipinitialspace=True)
        count = 0
        for row in whole_row_reader(read_iterator):
            if count == 0:
                count += 1
                continue
            angle_list.append(float(row[0]))
            Bairgap_list.append(float(row[1]))
    angle_list_first_half, angle_list_second_half = split_list(angle_list)
    Bairgap_list_first_half, Bairgap_list_second_half = split_list(Bairgap_list)

    # print len(angle_list_first_half), len(angle_list_second_half)
    # plt.plot(angle_list_second_half[:-1], Bairgap_list_second_half[:-1])

    plt.subplot2grid((3,2), (ind,1), colspan=2, rowspan=1)
    # plt.locator_params(axis='x', nbins=5)
    # plt.locator_params(axis='y', nbins=5)
    # plt.title('%gHz'%(freq))
    plt.xlabel('Air gap position [mech. deg] (at t=0.2 s)')
    plt.ylabel('Air gap flux density [T]')

    import operator
    def get_max_and_index(the_list):
        return max(enumerate(the_list), key=operator.itemgetter(1))

    plt.plot(angle_list_first_half, Bairgap_list_first_half,       'k--', label='%gHz (1st-half, 0-180 deg)'%(freq),  alpha=0.5, lw=0.75)
    plt.plot(angle_list_first_half, Bairgap_list_second_half[:-1], 'r-', label='%gHz (2nd-half, 180-360 deg)'%(freq), alpha=0.5, lw=0.75)
    plt.legend(loc='lower left')
    plt.grid()
    # plt.text(50, 1.2, 'Maximum difference is %.4f T.'% (max(abs(array(Bairgap_list_first_half) - array(Bairgap_list_second_half[:-1])))), fontsize=12)

    Bairgap_difference = abs(array(Bairgap_list_first_half) - array(Bairgap_list_second_half[:-1]))
    max_index, max_value = get_max_and_index(Bairgap_difference)
    plt.annotate(
        'Maximum difference is %.2f T.' % (max_value),
        xy=(angle_list_first_half[max_index], Bairgap_list_first_half[max_index]), arrowprops=dict(arrowstyle='->'), xytext=(115, 1.2),  fontsize=12)
    
    plt.xlim(0,180)
    plt.yticks(np.arange(-1.2, 1.21, 0.4))
    if ind==2:
        plt.xticks(np.arange(0, 181, 30), ['0\n180', '30\n210', '60\n240', '90\n270', '120\n300', '150\n330', '180\n360'])
    else:
        # plt.xticks(np.arange(0, 181, 30), ['']*7)
        plt.xticks([])

    b = int(120/180.*len(angle_list_first_half))
    e = int(135/180.*len(angle_list_first_half))
    locallyZoomIn(plt.gca(), [(angle_list_first_half[b:e], Bairgap_list_first_half[b:e]), 
                               (angle_list_first_half[b:e], Bairgap_list_second_half[b:e])], (-0.9,-0.3), loc1=1, loc2=3, zoom=2.0, loc='upper center')

    ind += 1



f_name4 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 0Hz-ForConAbs.csv'
f_name5 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 50Hz-ForConAbs.csv'
f_name6 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 500Hz-ForConAbs.csv'

plt.subplot2grid((6,2), (0,0), colspan=1, rowspan=3)
# plt.locator_params(axis='x', nbins=5)
# plt.locator_params(axis='y', nbins=5)
# plt.title('%gHz'%(freq))
plt.xlabel('Time [s]')
plt.ylabel('Force Amplitude [N]')
# plt.hold(True)

for f_name, freq in zip([f_name4, f_name5, f_name6], [7.5, 50, 500])[::-1]:
    time_list = []
    ForConAbs_list = []
    with open(f_name, 'r') as f:
        read_iterator = csv_reader(f, skipinitialspace=True)
        count = 0
        for row in whole_row_reader(read_iterator):
            if count == 0:
                count += 1
                continue
            time_list.append(float(row[0]))
            ForConAbs_list.append(float(row[1]))

    plt.plot(time_list, ForConAbs_list, label='%g Hz, %g rpm'%(freq, (freq-7.5)*30), alpha=0.7)
    plt.legend()
    plt.grid()
plt.xlim(0,0.2)
# plt.xticks(np.arange(0, 0.2, 0.05), ['']*len(np.arange(0, 0.2, 0.05)))
plt.xticks([])



f_name7 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 0Hz-TorCon.csv'
f_name8 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 50Hz-TorCon.csv'
f_name9 = r'D:\JMAG_Files\CSV\EC_Rotate_#0-Tran_#0 500Hz-TorCon.csv'


plt.subplot2grid((6,2), (3,0), colspan=1, rowspan=3)
# plt.locator_params(axis='x', nbins=5)
# plt.locator_params(axis='y', nbins=5)
# plt.title('%gHz'%(freq))
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
# plt.hold(True)

for f_name, freq in zip([f_name7, f_name8, f_name9], [7.5, 50, 500])[::-1]:
    time_list = []
    TorCon_list = []
    with open(f_name, 'r') as f:
        read_iterator = csv_reader(f, skipinitialspace=True)
        count = 0
        for row in whole_row_reader(read_iterator):
            if count == 0:
                count += 1
                continue
            time_list.append(float(row[0]))
            TorCon_list.append(float(row[1]))

    plt.plot(time_list, TorCon_list, label='%g Hz, %g rpm'%(freq, (freq-7.5)*30), alpha=0.7)
    plt.legend()
    plt.grid()



plt.text(0.03, 11, 'Maximum peak-to-peak torque ripple is below 1 Nm.', fontsize=12)
plt.xlim(0,0.2)
# plt.xticks(np.arange(0, 180, 30))



# fit subplots and save fig
fig.tight_layout()
# fig.set_size_inches(w=11,h=7)
fig_name = 'airgapBdifference500Hz50Hz7p5Hz.png'
fig.savefig(fig_name, dpi=150)

show()
quit()















# see also :https://stackoverflow.com/questions/28820618/pylab-adjust-hspace-for-some-of-the-subplots
if True:

    # https://scientificallysound.org/2016/06/09/matplotlib-how-to-plot-subplots-of-unequal-sizes/
    # Generate data
    dist_norm = np.random.normal(loc=0, scale=1, size=1000)
    dist_tdis = np.random.standard_t(df=29, size=1000)
    dist_fdis = np.random.f(dfnum=59, dfden=28, size=1000)
    dist_chsq = np.random.chisquare(df=2, size=1000)

    import matplotlib.gridspec as gridspec
    # Plot figure with subplots of different sizes
    fig = plt.figure(3)
    # set up subplot grid
    gridspec.GridSpec(3,3)

    # large subplot
    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.title('Normal distribution')
    plt.xlabel('Data values')
    plt.ylabel('Frequency')
    plt.hist(dist_norm, bins=30, color='0.30')

    # small subplot 1
    plt.subplot2grid((3,3), (0,2))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.title('t distribution')
    plt.xlabel('Data values')
    plt.ylabel('Frequency')
    plt.hist(dist_tdis, bins=30, color='b')

    # small subplot 2
    plt.subplot2grid((3,3), (1,2))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.title('F distribution')
    plt.xlabel('Data values')
    plt.ylabel('Frequency')
    plt.hist(dist_fdis, bins=30, color='r')

    # small subplot 3
    plt.subplot2grid((3,3), (2,2))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.title('Chi-square distribution')
    plt.xlabel('Data values')
    plt.ylabel('Frequency')
    plt.hist(dist_chsq, bins=30, color='g')

    # fit subplots and save fig
    fig.tight_layout()
    fig.set_size_inches(w=11,h=7)
    fig_name = 'plot.png'
    fig.savefig(fig_name)


    show()
