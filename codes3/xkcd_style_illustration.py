from pylab import *
from matplotlib.patches import Arc
import math

def get_angle_plot(line1, line2, offset = 1, color = None, origin = [0,0], len_x_axis = 1, len_y_axis = 1):
    l1xy = line1.get_xydata()

    # Angle between line1 and x-axis
    slope1 = (l1xy[1][1] - l1xy[0][1]) / float(l1xy[1][0] - l1xy[0][0])
    angle1 = abs(math.degrees(math.atan(slope1))) # Taking only the positive angle

    l2xy = line2.get_xydata()

    # Angle between line2 and x-axis
    slope2 = (l2xy[1][1] - l2xy[0][1]) / float(l2xy[1][0] - l2xy[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color() # Uses the color of line 1 if color parameter is not passed.

    return Arc(origin, len_x_axis*offset, len_y_axis*offset, 0, theta1, theta2, color=color, label = str(angle)+"\u00b0", zorder=99)
def get_angle_text(angle_plot, bool_show_value=False):
    if bool_show_value:
        angle = angle_plot.get_label()[:-1] # Excluding the degree symbol
        angle = "%0.2f"%float(angle)+"\u00b0" # Display angle upto 2 decimal places
    else:
        angle = "Force error angle $E_a$" # Display angle upto 2 decimal places

    # Get the vertices of the angle arc
    vertices = angle_plot.get_verts()
    print(vertices)

    # Get the midpoint of the arc extremes
    x_width = abs(vertices[0][0] + vertices[-1][0]) / 2.0 + 0.085
    y_width = abs(vertices[0][1] + vertices[-1][1]) / 2.0 - 0.1

    #print x_width, y_width

    separation_radius = max(x_width/2.0, y_width/2.0)

    return [ x_width + separation_radius, y_width + separation_radius, angle]       
# Force error calculation illustration plot 
with plt.xkcd():
# if True:
    # fig = figure(1, dpi=150, figsize=(16/2*2/3., 8/2), facecolor='w', edgecolor='k')
    fig, ax = subplots(1, 1, dpi=150)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('Y-aixs force')
    ax.set_xlabel('X-aixs force')
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 0.6])

    force_vectors = [[0.4525356 , 0.53021212],
                     [0.48239928, 0.42097418],
                     [0.33567002, 0.50421884],
                     [0.34091791, 0.54390709],
                     [0.43018469, 0.17636313]]
    lines = []
    for force_vector in force_vectors:
        lines += [ Line2D( [0, force_vector[0]],
                           [0, force_vector[1]], lw=0.1) ]
        ax.arrow(0,0, force_vector[0],force_vector[1], head_width=0.01)

    avg = [sum([el[0] for el in force_vectors])/5, sum([el[1] for el in force_vectors])/5]
    avg_line = Line2D( [0, avg[0]],
                       [0, avg[1]], lw=0.1) 
    ax.arrow(0,0, avg[0],avg[1], color='r', lw=2, head_width=0.02)
    ax.annotate(
        'Average force vector',
        xy=(avg[0]-0.035, avg[1]-0.04), arrowprops=dict(arrowstyle='->',color='r'), xytext=(avg[0]-0.1, avg[1]-0.2), color='r')

    ax.annotate(
        'The most deviated\nforce vector',
        xy=(0.43018469, 0.17636313), arrowprops=dict(arrowstyle='->',color='k'), xytext=(0.43018469-0.08, 0.17636313-0.12), color='k')

    angle_plot = get_angle_plot(avg_line, lines[-1], 0.3, color='b')
    angle_text = get_angle_text(angle_plot) 
    ax.add_patch(angle_plot) # To display the angle arc
    print(angle_text)
    ax.text(*angle_text, color='b')

fig_name = r'D:\OneDrive\[00]GetWorking\31 Bearingless_Induction_FEA_Model\p2019_iemdc_bearingless_induction full paper\images/' + 'XKCD_illustration_force_err_angle_Ea.png'
fig.savefig(fig_name, dpi=150)
show()  
quit()



from csv import reader as csv_reader
def whole_row_reader(reader):
    for row in reader:
        yield row[:]
def read_forceXY_data(XY='X'):
    f_name1 = 'D:/JMAG_Files/CSV/Severson02_EC_Rotate_PS_24cases-ForCon%s.csv'%(XY)
    freq_list = []
    ForConXY_list = []
    with open(f_name1, 'r') as f:
        read_iterator = csv_reader(f, skipinitialspace=True)
        count = 0
        for row in whole_row_reader(read_iterator):
            if count == 0:
                count += 1
                continue
            freq_list.append(float(row[0]))
            ForConXY_list.append([float(el) for el in row[1:]])

    ax.grid(True)
    ax.set_ylabel('%s-axis force [N]'%(XY))
    ax.set_xlabel('Slip frequency [Hz]')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])

    return freq_list, ForConXY_list
# Eddy current FEA with rotating rotor illustration plot 
with plt.xkcd():
    # fig = figure(1, dpi=150, figsize=(16/2*2/3., 8/2), facecolor='w', edgecolor='k')
    fig, ax_list = subplots(1, 2, dpi=150)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    ax = ax_list[0]
    freq_list, ForConX_list = read_forceXY_data(XY='X')
    loc_annotate = int(0.7*len(freq_list))
    lines = []
    for i in range(0, len(ForConX_list[0]), 8):
        y = [el[i] for el in ForConX_list]
        lines += ax.plot(freq_list, y, lw=1) # label=str(i*360./Qr/24.)
        ax.annotate(
            '$\\theta=%.0f^\\circ$'%(360/36./24.*i),
            xy=(freq_list[loc_annotate], y[loc_annotate]), arrowprops=dict(arrowstyle='->'), xytext=(freq_list[loc_annotate+1], y[loc_annotate]+0.5))
    ax.set_xlim([3.5,5])
    ax.set_ylim([145,162.5])
    breakdown_slip_freq = freq_list[loc_annotate]
    ax.plot([breakdown_slip_freq]*2, [145, 162.5], 'k--')

    ax = ax_list[1]
    freq_list, ForConY_list = read_forceXY_data(XY='Y')
    loc_annotate = int(0.6*len(freq_list))
    lines = []
    for i in range(0, len(ForConY_list[0]), 8):
        y = [el[i] for el in ForConY_list]
        lines += ax.plot(freq_list, y, lw=1) # label=str(i*360./Qr/24.)
        ax.annotate(
            '$\\theta=%.0f^\\circ$'%(360/36./24.*i),
            xy=(freq_list[loc_annotate], y[loc_annotate]), arrowprops=dict(arrowstyle='->'), xytext=(freq_list[loc_annotate+1], y[loc_annotate]+3.5))
    ax.set_xlim([3,5])
    ax.set_ylim([115,140])
    breakdown_slip_freq = freq_list[loc_annotate]
    ax.plot([breakdown_slip_freq]*2, [115, 140], 'k--')
# fig.tight_layout()
fig_name = r'D:\OneDrive\[00]GetWorking\31 Bearingless_Induction_FEA_Model\p2019_iemdc_bearingless_induction full paper\images/' + 'XKCD_illustration_ec_rotate.png'
fig.savefig(fig_name, dpi=150)
plt.show()

