# Code based on Python 3.x
# _*_ coding: utf-8 _*_
# Author: "LEMON"
# Website: http://liyangbit.com
# 公众号: Python数据之道
# ID: PyDataRoad

import plotly_express as px
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)

iris = px.data.iris()
iris_plot = px.parallel_coordinates(iris, color="species_id", labels={"species_id": "Species",
                  "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
                  "petal_width": "Petal Width", "petal_length": "Petal Length", },
                    color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2, template='plotly_dark')
# iris_plot = px.parallel_coordinates(iris, color='species_id', color_continuous_scale=['red', 'green', 'blue'])
# iris_plot = px.scatter_matrix(iris, dimensions=['sepal_width', 'sepal_length', 'petal_width', 'petal_length'], color='species', template='plotly_dark') # : plotly, plotly_white and plotly_dark
# iris_plot = px.scatter(iris, x='sepal_width', y='sepal_length',
#            color='species', marginal_y='histogram',
#           marginal_x='box', trendline='ols')
plotly.offline.plot(iris_plot)


tips = px.data.tips()
tips_plot = px.parallel_categories(tips, color='size', color_continuous_scale=px.colors.sequential.Inferno)
plotly.offline.plot(tips_plot)
    
quit()

# Code based on Python 3.x
# _*_ coding: utf-8 _*_
# Author: "LEMON"
# Website: http://liyangbit.com
# 公众号: Python数据之道
# ID: PyDataRoad

import plotly_express as px
import plotly
plotly.offline.init_notebook_mode(connected=True)

wind = px.data.wind()
wind_plot = px.bar_polar(wind, r="value", theta="direction", color="strength", template="plotly_dark",
            color_discrete_sequence= px.colors.sequential.Plotly[-2::-1])

plotly.offline.plot(wind_plot)



