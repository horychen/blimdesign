import matplotlib.pyplot as plt
# Pie chart
labels = ['Eddy Current', 'Hysteresis', 'Joule', 'Windage']

# % proj1130
# % Iron 1080.53, 914.927, 165.601
# % Copper 291.681, 168.433
# % Windage 413.42,
# % Total 1954.06
total_loss = 1954.06 # W
sizes = [914.927/total_loss, 165.601/total_loss, (291.681+168.433)/total_loss, 413.42/total_loss]

#colors # https://www.schemecolor.com/color/green
# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
# colors = ['#072F5F','#1261A0','#3895D3','#58CCED']
colors = ['#C766A1','#F49762','#FFEC8A','#A1D47B'] 

#explsion
explode = (0.025,0.025,0.025,0.025)


import matplotlib as mpl
mpl.rcParams['font.size'] = 15.0
mpl.rcParams['font.family'] = ['Times New Roman']
# font = {'family' : 'Times New Roman', #'serif',
#     'color' : 'darkblue',
#     'weight' : 'normal',
#     'size' : 14,}
plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.45, explode = explode)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
ax1 = plt.gca()
ax1.add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
# plt.tight_layout()
fig.savefig(r'C:\Users\horyc\Desktop/loss_donut_chart_proj1130.png')
plt.show()

# https://medium.com/@kvnamipara/a-better-visualisation-of-pie-charts-by-matplotlib-935b7667d77f

# 如果转速可以变化，可以画成Stack Plots
# https://www.youtube.com/watch?v=xN-Supd4H38
legend(loc=(0.05, 0.07))

