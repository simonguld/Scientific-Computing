import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from numpy import newaxis

from LJhelperfunctions import V,EPSILON,SIGMA,distance,gradV,flat_V,flat_gradV
NA = newaxis

data = np.load('ArStart.npz')
Xstart2 = data['Xstart2']
Xstart3 = data['Xstart3']
Xstart4 = data['Xstart4']
Xstart5 = data['Xstart5']
Xstart6 = data['Xstart6']
Xstart7 = data['Xstart7']
Xstart8 = data['Xstart8']
Xstart9 = data['Xstart9']
X20 = data['Xopt20']
Xopt20 = X20.reshape(-1)

# Equilibrium distance for potential
r0 = 2 ** (1 / 6) * SIGMA


### James' helper functions for making nice 3D plots:
def create_3d_plot():
    plot_dict = dict(projection='3d')
    fig, ax = plt.subplots(subplot_kw=plot_dict)
    return fig, ax
def make_axis_equal(ax):
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    z1, z2 = ax.get_zlim()
    m1 = min(x1, y1, z1)
    m2 = max(x2, y2, z2)
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)
    ax.set_zlim(m1, m2)
    ax.set_proj_type("ortho")
    return ax
def transparent_axis_background(ax):
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # Disable the axes
    ax.axis("off")
    return ax
def plot_min_distances(ax, points, cutoff=1.01 * r0):
    # points: (N,3)-array of (x,y,z) coordinates for N points
    # distance(points): returns (N,N)-array of inter-point distances:
    displacement = points[:, np.newaxis] - points[np.newaxis, :]
    r = np.sqrt(np.sum(displacement*displacement, axis=-1))
    # Grab only those that are in upper triangular part
    r = np.triu(r)
    # Grab lengths that are above 0, but below the cutoff
    mask = ( r <= cutoff) * (r > 0) * (0.99 * r0 < r)
    # Grab the indices of elements that satisfy the mask above
    ii, jj = np.where(mask)
    # For each pair of indices (corresponding to a distance close to optimal)
    # we plot a line between the corresponding points
    for i, j in zip(ii, jj):
        p = points[[i, j]]
        ax.plot(*p.T, color='k', ls="-")
    return ax
def create_base_plot(points):
    # Create the figure and 3D axis
    fig, ax = create_3d_plot()
    # translate points so centre of mass is at origin
    m = np.mean(points, axis=0)
    points = np.array(points) - m[np.newaxis, :]
    # Plot points and lines between points that are near to optimal
    ax.scatter(*points.T)
    ax = plot_min_distances(ax, points)
    return ax, points
def make_plot_pretty(ax):
    # Make the plot pretty
    ax = transparent_axis_background(ax)
    ax = make_axis_equal(ax)
    return ax





#SIMULATED ANNEALING

### Ting at pille ved:
# Bedre nabovalg
# fx #permutere hver vektor i en vilkårlig retningen i stedet for at angribe det koordinatvis.
# bruge et helt andet scheme??
#NBBBB! Perturbere med brownian/levy/gaussian i stedet for bare homogent random. Det fleste skridt tæt på,
# men med mulighed for store skridt. Opdater amplification og annealing scheme tilsvarende
# min damping på permutationen lige nu undertrykker måske for meget ved små temperaturer og spilder f-kald

# Andet cooling scheme
# Lege med T-afhængighed for cutoff for forskellige løsninger
# Overveje x_new opdatering når out of bounds. er random bedre en randen?
# gennemgå tingene igen. overveje, hvad der kan gøres klogere

def probability_function (fx, fx_new, T):
    assert fx_new >= fx
    if T == 0:
        return 0
    else:
        return np.exp(- (fx_new - fx) / T)
def random_neighbor_generator (x, iteration, scale):
    """
    This random neighbor generator calculates a random perturbation in
    [-scala/2, scala/2] and adds it to x[iteration]
    """
    perturbation = scale * np.random.rand(1) - scale/2

    perturbation_vector = np.zeros(np.size(x))
    perturbation_vector[iteration] = perturbation
    return x + perturbation_vector
def temperature_function (T_initial, T_steps, cycle_no, power = 1):
    """
    This simple cooling scale initializes the temperature as T_initial, and is then updated according to
    T_new = (1 - cycle_no / T_steps) * T,
    where T_steps is the maximal number of temperature changes, and cycle_no is the current number of changes.
    At the final step, T = 0, thus reducing the annealing scheme to a steepest descent so as to settle on a minimum
    in the final cycle.
    """

    T_new =  (1 - cycle_no / T_steps) ** power * T_initial
    return T_new
def simulated_annealing (f, x0, x_boundary, T_steps, T_initial, scale):
    """

    :param f: The object function to be minimized
    :param x: Flat input vector of length dim
    :param x_boundary: dim x 2 matrix defining the search space, satisfying x_boundary[i,:] = [xi_min,xi_max]
    :param T_steps: The total number of temperature changes before terminitation
    :param T_initial: Initial temperature
    :param scale: Determines the maximal perturbation size.
    The perturbations of each xi lie within [xi-scale/2,xi+scale/2].
    :return:
    """
    # Initialize
    dim = np.size(x0)
    iterations = 0
    cycle_no = 0
    scale_damping_attempt = 0
    scale_damping = np.linspace(1,dim,dim)
    tolerance = 1e-6

    T = T_initial
    x = x0.astype('float')
    fx = f(x)
    f_calls = 1
    f_best = fx
    x_best = x.astype('float')
    f_list = np.array(fx,'float')

    while cycle_no <= T_steps and iterations < dim:
        while iterations < dim:
            #restart
            if cycle_no/T_steps < 0.6 and T_steps % 40 == 0:
                x = x_best.astype('float')

            #x_new = random_neighbor_generator(x, iterations, 1 / scale_damping[scale_damping_attempt] * scale).astype('float')
            x_new = random_neighbor_generator(x, iterations, 0.3+((T_steps - cycle_no) / T_steps) ** 1 / scale_damping[scale_damping_attempt]).astype(
                'float')

            if x_new[iterations] < x_boundary[iterations, 0]:
                x_new[iterations] = x_boundary[iterations, 0]
            elif x_new[iterations] > x_boundary[iterations, 1]:
                x_new[iterations] = x_boundary[iterations, 1]

            fx_new = f(x_new)
            f_calls += 1

            if np.abs(fx-fx_new) > 3*T+5 and scale_damping_attempt<dim-1:
                scale_damping_attempt += 1
                continue
            else:
                scale_damping_attempt = 0
                iterations += 1
                if fx_new < fx:
                    np.append(f_list,fx_new)
                    fx = fx_new
                    x = x_new.astype('float')
                    if fx_new < f_best:
                        f_best = fx_new
                        x_best = x_new.astype('float')
                elif fx >= fx_new:
                    if probability_function(fx,fx_new,T) > np.random.rand(1):
                        np.append(f_list, fx_new)
                        fx = fx_new
                        x = x_new.astype('float')
       # if  (T_steps-cycle_no)/T_steps <= 0.1 and np.all(np.abs(f_list -fx) < tolerance) and np.abs(f_best-fx) < tolerance:
        #    print(f"Converged withtin {tolerance} after {f_calls} function calls ")
         #   break
        # update T and repeat
        cycle_no += 1
        T = temperature_function(T_initial,T_steps,cycle_no)
        iterations = 0

        f_list = []
    return x_best, f_best, f_calls


#GENETIC ALGORITHM
#fastfryse den del af sværmen, der er konvergeret og på den måde 'gro krystallen' / reducere dimensionaliteten løbende
#lave browninan / levy som mutation. evt mutere hele partikler og ikke koordinater?? men giver det mening?
def uniform_crossover_scheme (x1, x2, cross_over_rate,blending_step):
    """
    This scheme performs uniform crossover with blending (essentially coordinate averaging) every blending_step'th step
    Other blending schemes that allow for blended values to lie outside of the extreme values of x1 and x2 exist
    --> this necesitates an evaluation of whether new values lie within search space

    :param x1,x2: solutions to undergo crossover
    :param cross_over_rate: the rate at which crossover is carried out
    :param blending_step: Blending will occur at every blending_step'th step
    :return: x1_new, x2_new
    """

    #copy values
    x1_flat = x1.astype('float')
    x2_flat = x2.astype('float')
    x1_new = x1.astype('float')
    x2_new = x2.astype('float')

    dim = np.size(x1_new)

    #generate a random number in (0,1) for each variable:
    random_vector = np.random.rand(dim)
    #find indices where crossover occurs
    crossover_indices = np.argwhere(cross_over_rate > random_vector).flatten()

    # generate random starting index for blending
    blending_start_index = np.floor(np.random.rand(1) * dim)
    #find indices where blending occurs
    blend_indices = np.argwhere ( (crossover_indices+blending_start_index) % blending_step == 0 ).flatten()

    #delete the blending indices from crossover_indices
    crossover_indices = np.delete(crossover_indices,blend_indices)

    # perform crossover:
    x1_new[crossover_indices] = x2_flat[crossover_indices]
    x2_new[crossover_indices] = x1_flat[crossover_indices]

    # perform blending --> blending scheme ensures that new values are not out of bounds:
    beta = np.random.rand(1)    #beta could be chosen different for different coordinates
    x1_new[blend_indices] = beta * x1_flat[blend_indices] + (1 - beta) * x2_flat[blend_indices]
    x2_new[blend_indices] = beta * x2_flat[blend_indices] + (1 - beta) * x1_flat[blend_indices]

    return x1_new, x2_new
def mutation_scheme (X0, x_boundary, mutation_rate, mutation_scale, N_variables):
  """
  :param x: X is the collection of all solutions (x1,...,XN) in the form X[i,:] = xi, with x1 = x_best, which is saved
  from mutation. X is a N_population x N_var matrix
  :param x_boundary:
  :param mutation_rate:
  :param mutation_scale:
  :param N_variables: The number of variables in each solution
  :return: X
  """
  # Make a flattened copy of X
  X = X0.astype('float').flatten()
  dim = np.size(X)

  # save the best solution from mutation
  x_best = X[0:N_variables]
  # Consider only the remaining solutions
  X = X[N_variables::]
  dim = np.size(X)

  # generate an 1xdim matrix of random values called P. The indices where P <= mutation_rate will be the indices
  # where mutation will take place

  P = np.random.rand(dim)
  # array of indices where mutation will take place
  mutation_indices = np.argwhere(P < mutation_rate).flatten()

  # mutate values. Each value is perturbed by a random number between (-mutation_scale/2, mutation_scale)
  X[mutation_indices] =  X[mutation_indices] + mutation_scale * np.random.rand(np.size(mutation_indices))
  - 0.5 * mutation_scale #* np.random.rand(np.size(mutation_indices))

  # replace all values outside of the search space with a random number from [xmin,xmax].
  new_value_too_small_indices = np.argwhere(X < x_boundary[0]).flatten()
  new_value_too_large_indices = np.argwhere(X > x_boundary[1]).flatten()
  too_small_values = np.size(new_value_too_small_indices)
  too_large_values = np.size(new_value_too_large_indices)
  interval_distance = np.abs(x_boundary[1]-x_boundary[0])

  if too_small_values > 0:
      X[new_value_too_small_indices] = interval_distance * np.random.rand(too_small_values) - x_boundary[0]
  if too_large_values > 0:
      X[new_value_too_large_indices] = interval_distance * np.random.rand(too_large_values) - x_boundary[0]

  #add the best solution to X, ie send X --> [x_best,X]
  X = np.r_['0',x_best,X]
  #reshape and return
  return X.reshape(X0.shape)

# lave en fordeling omkring startpunkt i stedet for bare homogen tilfældighed som mutation
def genetic_algorithm (f, crossover_scheme, mutation_scheme, x_boundary, max_generations, mutation_scale, N_variables,
                       population_size = 10, tol = 1e-8, selection_rate = 0.5, mutation_rate = 0.20,
                       cross_over_rate = 0.5, blending_step = 5):
    """
noget med assume x_bounds bare et interval ie alle bounds ens
    :param selection_rate:
    :param f:
    :param x_boundary:
    :param crossover_scheme:
    :param mutation_scheme:
    :param max_generations:
    :param population_size:
    :param tol:
    :param mutation_rate:
    :param mutation_scale:
    :param cross_over_rate:
    :return: x, f_calls
    """
    #NB: One could include the inital point x in the inital population
    #NB: one could restrict the space so as not to be as square box
    #different crossover?
    # Should you only mutate children? Maybe mutate all but the best solution?
    # Should you throw worst solutions away instead of doing k matches? Simply keeping the best fraction
    # discards solutions of high energy potentially close to a minimum --> maybe it is too directional/punishing
    #include linesearch to find local optimum of each solution after mutation phase --> memetic algorithm
    #initalize through harmmonic oscillator
    #build into the problem the particle nature of the system, crossover or mutate partciles instead of components

    #Construct initial random population confined by the search space
    X = (x_boundary[1]-x_boundary[0]) * np.random.rand(population_size, N_variables) - x_boundary[0]*np.ones([population_size,N_variables])

    f_calls = 0
    generation_no = 0
    f_values = np.empty(population_size)

    while generation_no < max_generations:
        generation_no += 1
    #Step 1: Selection
        for i in np.arange(population_size):
            f_values[i] = f(X[i])

        f_calls += population_size

     #   print("cycleno, f val", generation_no, f_values)

        #find the indices of the function values from smallest to largest
        selection_index = np.argsort(f_values)
        #sort function values from smallest to largest
        f_values_sorted = f_values[selection_index]
        #Select the best function values to undergo crossover and discard the rest
        # find the index of the largest value of f chosen to undergo crossover
        solution_cutoff_index = int(np.floor(selection_rate*population_size))
     #   print(solution_cutoff_index, "cutind")
     #   print(X)

        #Sort the population according to their function value, from smallest to largest
        X = X[selection_index,:]        # the rows 0:solution_cutoff_index represent the selected solutions

      #  print ("\n \n \n X- sorted", X)
        #Record the best solution and its function value
        f_best = f_values_sorted[0]
        x_best = X[0,:]
        print("fest, xbest", f_best, "\n \n")
        #Step 2: Check convergence
        if np.all(np.abs(f_values_sorted[0:solution_cutoff_index] - f_best) < tol):
            print(f"GA converged in {f_calls} function calls")
            return x_best, f_calls

        #Step 3: Perform crossover
        # Having discarded population_size - solution_cutoff_index solutions-1, we need to generate this number of offspring
        offspring_no = population_size - solution_cutoff_index - 1
        crossover_no = int (offspring_no / 2)
      #  print (offspring_no, "offspring_no")
        #choose solutions pairs at random and pass them to the crossover_scheme to obtain two new solution for each pair
        for i in range(crossover_no):
            parent1_index = int ( np.floor (solution_cutoff_index * np.random.rand(1)))
            parent2_index = int ( np.floor (solution_cutoff_index * np.random.rand(1)))
            child1, child2 = crossover_scheme(X[parent1_index,:],X[parent2_index,:],cross_over_rate,blending_step)
            X[solution_cutoff_index+i, :] = child1
            X[solution_cutoff_index+i+1, :] = child2
       # print("\n \n X_cross_over", X)

        #Step 4: Mutation. Each coordinate in the offspring solutions is mutated with a random value defined
        # by the mutation scheme with probablity mutation_rate
        X[solution_cutoff_index:, :] = mutation_scheme(X[solution_cutoff_index:, :], x_boundary, mutation_rate, mutation_scale, N_variables)

    return x_best, f_calls


#CUCKOO SEARCH:
#fastfryse den del af sværmen, der er konvergeret og på den måde 'gro krystallen' / reducere dimensionaliteten løbende
def bisection_root(f, a, b, tolerance=1e-13):
    fa, fb = f(a), f(b)
    if np.sign(fa) == np.sign(fb):             # f(a) and f(b) must have opposite signs
        print('Bracket condition not met. Convergence not guaranteed')

    k, fm = 0, 1                                 # Count number of iterations. Set fm=1 to activate while loop
    iterations_max = 100

    while k < iterations_max:                # Continue until convergence criterion is met
        m = a + (b - a) / 2                    # Write midpoint like this to avoid ending up outside of interval
        k, fm = k+1, f(m)                        # Increase k, store f(m) and reuse to avoid additional evaluations of f
        if np.sign(fa) == np.sign(fm):          # If f(a) and f(m) has the same sign, there is a root in [m,b]
            a, fa = m, fm
        else:                                  # If f(b) and f(m) has the same sign, there is a root in [a,m]
            b, fb = m, fm
        times_called = 2+k                         #We call f two times initially and then 1 time per loop
        if np.abs(fm) < tolerance:
            return m, times_called

    print("not converged. None is returned")
    return
def levy_scale_finder (length_scale, fraction_contained, tol = 1e-2):
    """
    Given a Levy distribution with mean = 0, this function finds the scale alpha such that the probablity
    that a randomly chosen point x is contained within [0,length_scale] is given by fraction_contained up to
    the tolerance tol.
    :param length_scale: the maximum value of the interation range [0,end_point].
                    NB: if the domain of a variable x is [x_min,x_max], length_scale could be chosen as (x_max-x_min)/2,
                    such that if a coordinate is situated at the middle of the interval and allowed to Levy flight in
                    both directions, its new value will be within bounds p_fraction of the time
    :param fraction_contained:
    :param tol:
    :return: alpha
    """
    #define integration region
    range = np.linspace(0.01, np.abs(length_scale), 100)

    # we wish to find the root of the equation int(Levy(range,alpha))-fraction_contained. Define the objevtive function
    def root_function (scale):
        value = np.trapz(st.levy.pdf(range,0,scale),range)-fraction_contained
        return value

    #Make sure that the interval for alpha is large enough to ensure a bracket
    alpha_max = length_scale
    left_sign =  np.sign(root_function(0.01))
    iterations = 0
    while left_sign == np.sign(root_function(alpha_max)) and iterations < 10:
        iterations +=1
        alpha_max = 2 * alpha_max
    if left_sign == np.sign(root_function(alpha_max)):
        return 1 / length_scale
    else:
        alpha, f_calls = bisection_root(root_function,0.01, alpha_max,tolerance=tol)
        return alpha
def levy_flight(X0, x_boundary, scale):
    """
    Perform a levy flight on the point X0, i.e. permute each coordinate of X0 by a random number given by the Lévy
    distribtuion with mean = 0 and scale = scale. The scale is chosen s.t. a given fraction of permutations lie
    within the search space.
    If a permutation takes a coordinate value out of bounds, that value will instead be replaced with a random value
    in the search space
    :param X0: An MxN matrix of coordinates.
    :param x_boundary: interval [x_min,x_max]. It is assumed that all points have the same boundary, so that the
                       search space is a square box
    :param scale:
    :return: X # X - a matrix of new particle positions with the same shape as X0
    """
    #noget med -1 for retning

    # Initialize points and flatten
    X = X0.astype('float').flatten()
    dimension = np.size(X)

    #generate a number of random values, eg. one for each dimension,
    # as determined by the Lévy distribution with mean = 0 and scale factor = scale

    levy_values = st.levy.rvs(loc = 0, scale = scale, size = dimension)

    # generate random indices to determine which coordinate of X will be perturbed by which levy_value
    random_indices = np.floor(np.random.rand(dimension) * dimension).astype('int')

    # generate random numbers 0,1 for each coordinate to determine whether to add or subtract the levy_value
    random_sign = np.random.randint(0, 2, dimension)
    #perform Levy flight
    X[random_indices] = X[random_indices] + (-1)**random_sign * levy_values

    # replace all values outside of the search space with a random number from [xmin,xmax].
    new_value_too_small_indices = np.argwhere(X < x_boundary[0]).flatten()
    new_value_too_large_indices = np.argwhere(X > x_boundary[1]).flatten()
    too_small_values = np.size(new_value_too_small_indices)
    too_large_values = np.size(new_value_too_large_indices)
    interval_distance = np.abs(x_boundary[1] - x_boundary[0])

    if too_small_values > 0:
        X[new_value_too_small_indices] = interval_distance * np.random.rand(too_small_values) - x_boundary[0]
    if too_large_values > 0:
        X[new_value_too_large_indices] = interval_distance * np.random.rand(too_large_values) - x_boundary[0]

    #reshape to original shape and return
    return X.reshape(X0.shape)
#improvements:
# make scale dependent on generation number. It seems that convergence slows down as max_gen is increased,
# indicating that something should be done about the amplification to make it invariant under this change
# use array programming to discard the cuckoo for-loop.
def levy_flight_cuckoo_search (f, x_boundary, max_generations, population_size, N_variables, discard_fraction = 0.25,
                               fraction_contained = 0.8, tol = 1e-8):
    """
    :param f: objective function
    :param x_boundary: We assume identicaly bounaries for all particles, so that x_boundary = [x_min,x_max], and
                       the search space is a square box
    :param max_generations: The maximum number of generation run before termination
    :param population_size: Number of solutions keps at any time
    :param discard_fraction:  fraction of the worst solutions that are discarded at the end of each generation
    :param fraction_contained: fraction of Levy steps that will be contained within [x_min,xmax] for a variable located
                            in the middle of the interval
    :param tol: The search terminates if the (1-discard_fraction) best solutions all lie within tol of the best solution
    :return: x_best, fcalls
    """
    # initalize:

    #find scale s.t. fraction_contained of all Levy flights will generate a new value within bounds if the starting
    # point is located in the middle of the interval [xmin,xmax]
    interval_distance = np.abs(x_boundary[1] - x_boundary[0])
    scale = levy_scale_finder(interval_distance / 2, fraction_contained)

    f_calls, generation_no = 0, 0

    # generate population_size random points in the search space, in the format of an population_size x N_variables matrix
    X = interval_distance * np.random.rand(population_size, N_variables) - x_boundary[0] * np.ones([population_size, N_variables])

    # calculate function values
    f_values = np.empty(population_size)
    for i in range(population_size):
        f_values[i] = f(X[i,:])
    f_calls += population_size

    # find index of best/smallest value of f
    index_best = np.argmin(f_values)

    #record best value of f and x
    f_best = f_values[index_best]
    x_best = X[index_best,:]

    #continue for at most max generations or until convergence is met
    while generation_no < max_generations:
        if generation_no % 100 == 0:
            print(generation_no)
        generation_no += 1

        #STEP 1: generate a cuckoo from each solution of the population, compare it to an arbitrary nest, and replace it
        #if a better solution is found
        for i in range(population_size):
            supression = 0.05 + (max_generations-generation_no) ** 0.5 /max_generations
            cuckoo = levy_flight(X[i,:], x_boundary, scale * supression)
            f_cuckoo = f(cuckoo)

            #compare quality of cuckoo solution to a random solution in the population
            random_index = int(np.floor(population_size*np.random.rand(1)))
            if f_values[random_index] > f_cuckoo > f_best:
                f_values[random_index] = f_cuckoo
                X[random_index,:] = cuckoo
            elif f_best > f_cuckoo:
                f_values[index_best] = f_cuckoo
                X[index_best,:] = cuckoo
        f_calls += population_size

        #STEP 2: Discard discard_fraction of worst solutions and replace them with new arbitray ones
        keep_fraction = 1 - discard_fraction
        #index for the number of solutions to keep, indexed from 0 so that keep_solution = 1 means keep first 2 solutions
        keep_index = np.floor(keep_fraction * N_variables).astype('int')

        #sort function values from best to worst / smallest to largest and obtain indices
        sorted_index = np.argsort(f_values)
        # find indices of values to be replaced
        discard_index = sorted_index[keep_index::]
        # record the number of values to be discarded
        no_discarded_values = np.size(discard_index)

        #construct new solutions
        X[discard_index, : ] = interval_distance * np.random.rand(no_discarded_values, N_variables)\
                              - x_boundary[0] * np.ones([no_discarded_values, N_variables])

        #calculate corresponding function values
        for i in range(no_discarded_values):
            f_values[discard_index[i]] = f(X[discard_index[i], :])
            f_calls += 1

        #update f_best and x_best
        index_best = np.argmin(f_values)
        f_best = f_values[index_best]
        x_best = X[index_best,:]

        print(f_best)
        #Step 3: Stop if the keep_fraction best fraction of solutions hve converged to f_best within tol
        #OBS: Including this step might cause the search to terminate at a local minimum
        if np.all(np.abs(f_best-f_values[0:keep_index]) < tol):
            print(f"converged to {tol} within {f_calls} function calls")
            return x_best, f_calls

    return x_best, f_calls

#Firefly algorithm:
def brownian_motion (X0, x_boundary, scale):
    """
    Takes an MxN - of a 1xMN matrix as input, and perturbs each entry with a random value determined by a normal distribution
    with mean = 0 and standard deviation = scale / 2, such that 96% of the magnitude of all perturbations will be
    less than scale. If a perturbation takes a coordinate value of out bounds, it will be replaced by a random value
    in the interval [xmin,xmax] instead.
    :param X0: MxN matrix to be perturbed
    :param x_boundary: an interval [xmin,xmax] defining the domain of each entry. The domain is assumed to be the same
    for all coordinates
    :param scale: a problem specific scale
    :return: X - a matrix of new particle positions with the same shape as X0
    """
    #flatten and copy
    X = X0.astype('float').flatten()
    dimension = np.size(X)
    #generate perturbations as random values of a normal distribution with mean = 0 and std = scale/2
    perturbation = st.norm.rvs(size = dimension, loc = 0, scale = scale / 2)

    #add perturbations to current coordinate values
    X = X + perturbation

    #find indices of values out of bounds and replace them with a random number in [x_boundary[0],x_boundary[1]]
    too_small_indices = np.argwhere( X < x_boundary[0]).flatten()
    too_large_indices = np.argwhere( X > x_boundary[1]).flatten()
    no_too_small = np.size(too_small_indices)
    no_too_large = np.size(too_large_indices)

    #replace out of bounds values
    X[too_small_indices] = (x_boundary[1]-x_boundary[0]) * np.random.rand(no_too_small) - x_boundary[0]
    X[too_large_indices] = (x_boundary[1]-x_boundary[0]) * np.random.rand(no_too_large) - x_boundary[0]

    #reshape to original shape
    return X.reshape(X0.shape)
def firefly_algorithm(f, perturbation_scheme, x_boundary, absorbtion, attraction_scale, attraction_exponent, max_iterations, N_variables, population_size, tol = 1e-8):
    """

    :param X0:
    :param perturbation_scheme: a scheme that takes X0 and perturbs each entry according to some distribution
    :param x_boundary: [xmin,xmax], the boundaries of each coordinate entry. The are assumed to be the same for all
    coordinates, so that the search space is a bax
    :param absorbtion: the absorbtion coefficient that determines how rapidly the attraction between particles decays
    with length
    :param attraction_scale: determines the prefactor of the attraction function
    :param attraction_exponent: the value of the exponent of the value difference between two solutions.
                                The attraction is scaled with this difference, so that very good solutions attract more
                                than mediocre solutions at the same distance - how much is determined by this parameter
    :param max_generations: the maximum number of iterations before termination
    :param N_variables: the number of coordinates in each vector/particle
    :param population_size: the number of particles the swarm / population
    :param tol: a convergence parameter, termination will occur if all values of f differ by no more than tol
    :return: X0, f_calls
    """
    # generate a swarm of particles taking random values in [xmin,xmax]
    X = (x_boundary[1] - x_boundary[0]) * np.random.rand(population_size, N_variables) - x_boundary[0]

    f_calls = 0
    iterations = 0
    f_values = np.empty(population_size)

    while iterations < max_iterations:
        iterations += 1

        #contstruct an population_size X population_size X 3 matrix such that distances_ijk = x_jk - x_ik
        rel_positions = -1 * ( X[:,np.newaxis] - X[np.newaxis,:] )

        #evalute function values
        for i in range(population_size):
            f_values[i] = f(X[i,:])
        f_calls += population_size

        #find the leader and record values
        if iterations == 1:
            f_best_index = np.argmin(f_values).flatten()
            f_best = f_values[f_best_index]
            x_best = X[f_best_index,:]
        else:
            f_current_best_index = np.argmin(f_values)
            if f_values[f_current_best_index] < f_best:
                f_best = f_values[f_current_best_index]
                x_best = X[f_current_best_index,:]
            #if the old f_best is the best solution, the perturbation made the solution worse, and we reinstate the old leader
            else:
                f_values[f_current_best_index] = f_best
                X[f_current_best_index,:] = x_best

        # construct function difference matrix st. Fij = f(xj) - f(xi)
        F = f_values[np.newaxis, :] - f_values[:, np.newaxis]

        # construct a 'follow matrix' such that follow_ij = 0 if Fj>=Fi and 1 otherwise.
        # this is useful since particle i will only be attracted to particle j if Fj<Fi
        follow = 0.5 * (1 - np.copysign(1, F))

        # construct an attraction matrix s.t. alpha_ij = attraction_scale * (F(xj) - F(xi) ** a * exp (- absortion * |dij|^2)
        #we first define the distance matrix st distance_ij = r_ij
        distances = np.sqrt(np.sum(rel_positions * rel_positions, axis = 2))
        #determine the exponent of distances in the exponential
        power = 2
        attraction = attraction_scale * np.power(F, attraction_exponent) * np.exp(-absorbtion * np.power(distances, power))

        #calculate free will perturbations to all coordinates with scale = (x_boundary[1]-x_boundary[0] ) / 4, ie
        # st 96% of all perturbations will lie within +/- scale rel. to the starting point.
        # dampen perturbations as max_generations is reached.
        damping_scheme = (max_iterations - iterations) / max_iterations
        damping_exp = 4
        #add a small constant to allow a small random walk even towards the end
        scale = 1e-10 + np.power(damping_scheme, damping_exp) * ( x_boundary[1] - x_boundary[0] ) / 3
        #X_pertubed is the perturbed values
        X_perturbed = perturbation_scheme(X, x_boundary, scale).astype('float')

        #calculate the directions of flight, i.e. the changes in the particle positions due to attraction and free will
        X = X_perturbed + np.sum(attraction[:,:,NA] * follow[:,:,NA] * rel_positions, axis = 1)

        if iterations == max_iterations - 1:
            for i in range(population_size):
                f_values[i] = f(X[i, :])
            f_calls += population_size
            #record the best value in the new population
            f_current_best_index = np.argmin(f_values)
            #compare the new best values with the best value before updating and return the best solution
            if f_values[f_current_best_index] < f_best:
                x_best = X[f_current_best_index,:]

    return x_best, f_calls


if 1:
    dim1 = 2.2 * r0 * np.ones(5)
    dim2 = 2.7 * r0 * np.ones(3)
    dim3 = 3.5 * r0
    dimensions = np.r_['0', dim1, dim2, dim3]

    particle_numbers = np.array([2, 3, 4, 5, 6, 7, 8, 9, 20])
    # dimension = 0.5*particle_numbers*r0

    N = 9
    fraction = 0.993
    index = np.argwhere(particle_numbers == N).flatten()
    interval = np.r_['0', np.zeros(1), dimensions[index]]

    x_best, f_calls = firefly_algorithm(flat_V,brownian_motion,interval, absorbtion=0.1, attraction_scale= 5,
                                        attraction_exponent = 1, max_iterations = 2000, N_variables= 3*N,population_size=35,
                                        tol = 1e-8)

    print(flat_V(x_best), f_calls)
    X = x_best.reshape(-1, 3)
    av_distance = np.sum(distance(X)) / (N ** 2 - N)
    # Calculate average distances in units of equilibrium distance r0
    av_distance_unit_r0 = av_distance / r0
    # Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
    ideal_distances = 0.5 * np.sum(np.abs((distance(X) - r0) / r0) <= 0.01)

    print(f"For {N} particles: ", f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ",
          f" "
          f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, ",
          f" # of functions calls = {f_calls}")

    ax, points = create_base_plot(x_best.reshape(-1, 3))
    ax = make_plot_pretty(ax)
    plt.title(f"Minimum energy configuration of {N} Argon atoms", fontsize=15)
    plt.show()


if 0:
    x_max = 2.5 * r0

    x_boundary = np.array([0, x_max])
    X = brownian_motion(Xstart4, x_boundary, x_max/2)
    print(Xstart4)
    print("\n\n", X)




if 0:
    dim1 = 2.6*r0*np.ones(5)
    dim2 = 3*r0*np.ones(3)
    dim3 = 3.5*r0
    dimensions = np.r_['0',dim1,dim2,dim3]


    particle_numbers = np.array([2,3,4,5,6,7,8,9,20])
    #dimension = 0.5*particle_numbers*r0

    N = 9
    fraction = 0.993
    index = np.argwhere(particle_numbers == N).flatten()
    interval = np.r_['0', np.zeros(1), dimensions[index]]

    x_best,f_calls = levy_flight_cuckoo_search(flat_V,interval,max_generations=1000,population_size=25, N_variables=3*N,
                                               discard_fraction=0.25, fraction_contained=fraction, tol=1e-5)

    print(flat_V(x_best),f_calls)
    X = x_best.reshape(-1,3)
    av_distance = np.sum(distance(X)) / (N ** 2 - N)
    # Calculate average distances in units of equilibrium distance r0
    av_distance_unit_r0 = av_distance / r0
    # Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
    ideal_distances = 0.5 * np.sum(np.abs((distance(X)-r0)/r0)<=0.01)

    print(f"For {N} particles: ", f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
    f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, " , f" # of functions calls = {f_calls}")


    ax, points = create_base_plot(x_best.reshape(-1, 3))
    ax = make_plot_pretty(ax)
    plt.title(f"Minimum energy configuration of {N} Argon atoms", fontsize=15)
    plt.show()





if 0:
    x_max = 2.5*r0
    fraction = 0.9

    alpha = levy_scale_finder(x_max/2,fraction)
    print(alpha)
    x_boundary = np.array([0,x_max])
    X = levy_flight(Xstart4,x_boundary,alpha)
    print(Xstart4)
    print("\n\n", X)

if 0:
    print(alpha)
    end_point = r0
    range = np.linspace(0.01,np.abs(end_point))
    print(np.trapz(st.levy.pdf(range,0,alpha),range))

    s = st.levy.rvs(0,alpha,1000)
    plt.hist(s,bins=1000, range =(0,2*end_point))
    plt.show()


if 0:
    dim1 = 2.5*r0*np.ones(5)
    dim2 = 3*r0*np.ones(3)
    dim3 = 3*r0
    dimensions = np.r_['0',dim1,dim2,dim3]

    particle_numbers = np.array([2,3,4,5,6,7,8,9,20])
    #dimension = 0.5*particle_numbers*r0

    N = 20
    index = np.argwhere(particle_numbers == N).flatten()

    # scalar 0.22 good for N=20
    #dimensions = 0.7 * 1/np.sqrt(2)*(N-1)*
    #dimensions = 3*r0
    #dimensions = 1*(N-1)*r0
    #dimensions = 2*(N-1)* r0
    interval = np.array([0, dimensions[index]])
    x_bound = np.ones([3*N,2])*interval

if 0:
    x_best, f_calls = genetic_algorithm(flat_V, uniform_crossover_scheme, mutation_scheme, interval, max_generations=1e4,
                                        mutation_scale=dimensions[index]/(N+N/2), N_variables = 3 * N,population_size=50,
                                        tol=1e-8,selection_rate=0.5, mutation_rate=0.15,cross_over_rate=0.7,blending_step=3)
    print(x_best,f_calls)
    X = x_best.reshape(-1,3)
    av_distance = np.sum(distance(X)) / (N ** 2 - N)
    # Calculate average distances in units of equilibrium distance r0
    av_distance_unit_r0 = av_distance / r0
    # Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
    ideal_distances = 0.5 * np.sum(np.abs((distance(X)-r0)/r0)<=0.01)

    print(f"For {N} particles: ", f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
    f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, " , f" # of functions calls = {f_calls}")


    ax, points = create_base_plot(x_best.reshape(-1, 3))
    ax = make_plot_pretty(ax)
    plt.title(f"Minimum energy configuration of {N} Argon atoms", fontsize=15)
    plt.show()



if 0:
    x_best, f_calls = genetic_algorithm(flat_V, uniform_crossover_scheme, mutation_scheme, interval, max_generations=1e4,
                                        mutation_scale=dimensions[index]/(N+N/2), N_variables= 3 * N,population_size=50,tol=1e-8
                                        ,selection_rate=0.5, mutation_rate=0.15,cross_over_rate=0.7,blending_step=3)
    print(x_best,f_calls)
    X = x_best.reshape(-1,3)
    av_distance = np.sum(distance(X)) / (N ** 2 - N)
    # Calculate average distances in units of equilibrium distance r0
    av_distance_unit_r0 = av_distance / r0
    # Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
    ideal_distances = 0.5 * np.sum(np.abs((distance(X)-r0)/r0)<=0.01)

    print(f"For {N} particles: ", f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
    f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, " , f" # of functions calls = {f_calls}")


    ax, points = create_base_plot(x_best.reshape(-1, 3))
    ax = make_plot_pretty(ax)
    plt.title(f"Minimum energy configuration of {N} Argon atoms", fontsize=15)
    plt.show()



if 0:
    X = mutation_scheme(Xstart3,[0,dimensions],0.2,dimensions,N)
    print("\n \n", Xstart3,"\n \n")
    print (X)

if 0:
    x1 = Xstart2[0:3]
    x2 = Xstart2[3::]

    print (x1, "    ", x2)
    x1_new, x2_new = crossover_scheme(x1,x2,0.5,2)
    print (x1_new, "    ", x2_new)

    print()

if 0:
    X2 = dimensions * np.random.rand(6)
    x1_new, x2_new = crossover_scheme(Xstart2, X2, 0.5, 2)
    print(Xstart2, "\n", X2)
    print("\n \n", x1_new, "\n", x2_new)

N=20
dim = 0.7 * 1/np.sqrt(2)*(N-1)
x_bound = np.ones([3*N,2])*np.array([0,dim])
if 0:
    x, f, calls = simulated_annealing(flat_V,Xopt20,x_bound,T_steps=1e3,T_initial=9e3,scale=7)

    print(x, f, calls)
    ax,points = create_base_plot(x.reshape(-1,3))
    ax = make_plot_pretty(ax)
    plt.title(f"Minimum energy configuration of {N} Argon atoms", fontsize = 15)
    plt.show()
