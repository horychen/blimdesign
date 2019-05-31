"""
Created on Wed May 29 22:46:00 2019
"""
from pylab import plt, np

class FEA_Solver:
    def __init__(self):
        pass

def my_plot(fits, vectors, ndf):

    if True:
        for fit in fits:
            print(fit)
        # ax = pg.plot_non_dominated_fronts(fits)
        ax = pg.plot_non_dominated_fronts(fits, comp=[0,1], marker='o')
        # ax = pg.plot_non_dominated_fronts(fits, comp=[0,2], marker='o')
        # ax = pg.plot_non_dominated_fronts(fits, comp=[1,2], marker='o')
    else:
        print('Valid for 2D objective function space only for now.')
        fig, axes = plt.subplots(ncols=2, nrows=1, dpi=150, facecolor='w', edgecolor='k');
        ax = axes[0]

        for index, xy in enumerate(vectors):
            ax.plot(*xy, 's', color='0')
            ax.text(*xy, '#%d'%(index), color='0')
        ax.title.set_text("Decision Vector Space")

        ax = axes[1]

        for index, front in enumerate(ndf):
            print('Rank/Tier', index, front)
            the_color = '%g'%(index/len(ndf))
            for individual in front: # individual is integer here
                ax.plot(*fits[individual], 'o',                 color=the_color)
                ax.text(*fits[individual], '#%d'%(individual),  color=the_color)

            front = front.tolist()
            front.sort(key = lambda individual: fits[individual][1]) # sort by y-axis value # individual is integer here
            for individual_A, individual_B in zip(front[:-1], front[1:]): # front is already sorted
                fits_A = fits[individual_A]
                fits_B = fits[individual_B]
                ax.plot([fits_A[0], fits_B[0]], 
                        [fits_B[1], fits_B[1]], color=the_color, lw=0.25) # 取高的点的Y
                ax.plot([fits_A[0], fits_A[0]], 
                        [fits_A[1], fits_B[1]], color=the_color, lw=0.25) # 取低的点的X

    ax.title.set_text("Pareto Front | Objective Function Space")

def my_print(ndf, dl, dc, ndr):
    # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
    # extract and print non-dominated fronts
    # - ndf (list of 1D NumPy int array): the non dominated fronts
    # - dl (list of 1D NumPy int array): the domination list
    # - dc (1D NumPy int array): the domination count
    # - ndr (1D NumPy int array): the non domination ranks

    for index, front in enumerate(ndf):
        print('Rank/Tier', index, front)
    count = 0
    for domination_list, domination_count, non_domination_rank in zip(dl, dc, ndr):
        print('Individual #%d\t'%(count), 'Belong to Rank #%d\t'%(non_domination_rank), 'Dominating', domination_count, 'and they are', domination_list)
        count += 1

import pygmo as pg
class Problem_BearinglessInductionDesign:
    def __init__(self):
        self.counter_fitness_called = 0
        self.counter_fitness_return = 0
    # Define objectives
    def fitness(self, x):
        self.counter_fitness_called += 1

        # denormalize the chromosome
        min_b, max_b = np.asarray(self.spec.bounds_denorm).T 
        diff = np.fabs(min_b - max_b)
        x_denorm = min_b + x * diff



        # Torque
        f1 = (x[0])**2
        # Efficiency @ rated power
        f2 = (x[1] + x[2])**2
        # Ripple performance
        f3 = (x[2] + x[])**2

        self.counter_fitness_return += 1
        return [f1, f2, f3]

    # Return number of objectives
    def get_nobj(self):
        return 3

    # Return bounds of decision variables (a.k.a. chromosome)
    def get_bounds(self):
        return ([0]*7, [1]*7)

    # Return function name
    def get_name(self):
        return "MySchaffer function"

    # Some common properties of the electric machine are given here
    def init_machine_template(self, spec):
        self.spec = spec

# create UserDefinedProblem and create population
prob = pg.problem(Problem_BearinglessInductionDesign())
pop = pg.population(prob, size=36)
print('-'*40, '\n', pop)

# select algorithm (another option is pg.nsga2(gen=1))
algo = pg.algorithm(pg.moead(gen = 1, weight_generation = "grid", decomposition = "tchebycheff", 
                                neighbours = 20, CR = 1, F = 0.799999, eta_m = 20, realb = 0.9, limit = 2, 
                                preserve_diversity = True)) # https://esa.github.io/pagmo2/docs/python/algorithms/py_algorithms.html#pygmo.moead
print('-'*40, '\n', algo)

# Begin optimization
number_of_iterations = 2
for _ in range(number_of_iterations):
    pop = algo.evolve(pop)
    fits, vectors = pop.get_f(), pop.get_x()
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
    my_print(ndf, dl, dc, ndr)
    my_plot(fits, vectors, ndf)


# # select algorithm
# algo = pg.algorithm(pg.moead(gen=500))
# print('-'*40, '\n', algo)

# xs = [fit[0] for fit in fits]
# ys = [fit[1] for fit in fits]
# zs = [fit[2] for fit in fits]
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(xs, ys, zs, marker='o')
# plt.show()

