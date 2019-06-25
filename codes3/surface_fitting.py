import numpy, scipy
import scipy.optimize
import matplotlib
from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm # to colormap 3D surfaces from blue to red
import matplotlib.pyplot as plt

graphWidth = 800 # units are pixels
graphHeight = 600 # units are pixels

# 3D contour plot lines
numberOfContourLines = 16


def SurfacePlot(equationFunc, data, params):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)

    matplotlib.pyplot.grid(True)
    axes = Axes3D(f)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    Z = equationFunc(numpy.array([X, Y]), *params)

    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)

    axes.scatter(x_data, y_data, z_data) # show data along with plotted surface

    axes.set_title('Surface Plot (click-drag with mouse)') # add a title for surface plot
    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label
    axes.set_zlabel('Z Data') # Z axis data label

    plt.show()
    plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems


def ContourPlot(equationFunc, data, params):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    Z = equationFunc(numpy.array([X, Y]), *params)

    axes.plot(x_data, y_data, 'o')

    axes.set_title('Contour Plot') # add a title for contour plot
    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label

    CS = matplotlib.pyplot.contour(X, Y, Z, numberOfContourLines, colors='k')
    matplotlib.pyplot.clabel(CS, inline=1, fontsize=10) # labels for contours

    plt.show()
    plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems


def ScatterPlot(data, title):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)

    matplotlib.pyplot.grid(True)
    axes = Axes3D(f)
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    axes.scatter(x_data, y_data, z_data, depthshade=False, color='k')

    axes.set_title(title)
    axes.set_xlabel('X Data')
    axes.set_ylabel('Y Data')
    axes.set_zlabel('Z Data')

    plt.show()
    plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems


def EquationFunc(data, *params):
    p0 = params[0]
    p1 = params[1]
    return p0 + numpy.sqrt(data[0]) + numpy.cos(data[1] / p1)

def surface_fitting(xData, yData, zData):

    pInitial = (1.0, 1.0)
    popt, pcov = scipy.optimize.curve_fit(EquationFunc,(xData,yData),zData, p0=pInitial)

    dataForPlotting = [xData, yData, zData]

    ScatterPlot([xData, yData, zData], 'Data Scatter Plot (click-drag with mouse)')
    SurfacePlot(EquationFunc, [xData, yData, zData], popt)
    ContourPlot(EquationFunc, [xData, yData, zData], popt)

    # absError = zData - EquationFunc((xData,yData), *popt)
    # ScatterPlot([xData, yData, absError], 'Error Scatter Plot (click-drag with mouse)')

if __name__ == "__main__":

    # raw data
    xData = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    yData = numpy.array([11.0, 12.1, 13.0, 14.1, 15.0, 16.1, 17.0, 18.1, 90.0])
    zData = numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.0, 9.9])

    pInitial = (1.0, 1.0)
    popt, pcov = scipy.optimize.curve_fit(EquationFunc,(xData,yData),zData, p0=pInitial)

    dataForPlotting = [xData, yData, zData]

    ScatterPlot([xData, yData, zData], 'Data Scatter Plot (click-drag with mouse)')
    SurfacePlot(EquationFunc, [xData, yData, zData], popt)
    ContourPlot(EquationFunc, [xData, yData, zData], popt)

    # absError = zData - EquationFunc((xData,yData), *popt)
    # ScatterPlot([xData, yData, absError], 'Error Scatter Plot (click-drag with mouse)')


