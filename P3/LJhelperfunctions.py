import numpy as np
from numpy import newaxis, fill_diagonal, sum, sqrt
NA = newaxis

# Experimental values for Argon atoms
EPSILON=0.997 # kJ/mol
SIGMA=  3.401 # Ångstrom
r0 = 2 ** (1 / 6) * SIGMA

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


def distance(points):
    # points: (N,3)-array of (x,y,z) coordinates for N points
    # distance(points): returns (N,N)-array of inter-point distances:

    ## points[NA,:] creates a 1,N,3 array of points, i.e. copies of the matrix points along first axis points[NA,:] = points[NA,N,3]=[points,ponts,points,...]
    ## points[:,NA] creates an N,1,3 array. points[:,NA]=points[N,NA,3] = [x1, x2, ..., xn]
    ## subtracting the two makes a N,N,3 matrix, where each copy of points is subtracted by each row of x1
    ## that is, for points[:,NA] = points[:,NA,:], we will copy along the rows. So row one, x1, will subtracted from the
    ## 1st NA entry in points[NA,:], which is just points, yielding points-x1*Identity.

    ## ie displacement = (points-x1*Id,points-x2*Id,...,points-x_n*Id) = ([x1-x1;x2-x1,...,xN-x1],[x1-x2;x2-x2,...xN-xn],...,[x1-xN;...;xN-xN]
    ## summing displacement*2 over axis -1 (i.e last row ie the columns) yields then an NxN matrix
    ## sum (discplacement**2,axis=2) = [(x1-x1)^2, (x1-x2)^2, ... (x1-xN)^2; (x2-x1)^2,(x2-x1)^2,...
    ## this producing an NxN matrix of distances between all pairs of particles. Only Upper half contains non-redundant info

    displacement = points[:, NA] - points[NA, :]
    return sqrt(sum(displacement*displacement, axis=-1))


def LJ(sigma=SIGMA, epsilon=EPSILON):
    def V(points):
        # points: (N,3)-array of (x,y,z) coordinates for N points
        dist = distance(points)

        # Fill diagonal with 1, so we don't divide by zero
        fill_diagonal(dist, 1)

        # dimensionless reciprocal distance
        f = sigma/dist

        # calculate the interatomic potentials
        pot = 4*epsilon*(f**12 - f**6)

        # Undo any diagonal terms (the particles don't interact with themselves)
        fill_diagonal(pot, 0)

        # We could return the sum of just the upper triangular part, corresponding
        # to equation 2 in the assignment text. But since the matrix is symmetric,
        # we can just sum over all elements and divide by 2
        return sum(pot)/2
    return V


def LJgradient(sigma=SIGMA, epsilon=EPSILON):
    def gradV(X):
        d = X[:, NA] - X[NA, :]
        r = sqrt(sum(d*d, axis=-1))
        ## r is just equal to the distance matrix.
        fill_diagonal(r, 1)

        T = 6*(sigma**6)*(r**-7)-12*(sigma**12)*(r**-13)
        # T is an (N,N)−matrix of r−derivatives. the entry ij gives the size!! of gradient between particles i and j.
        # so essentially T collects the size of the forces with T_ij being the negative force being particles i and j


        # this makes sense since the force is concerced only with rel. distances. to find total force on xi, one simply sum contributions
        # we would like to rewrite this to an N,3 matrix giving the gradient at each gradV = (gradV(x1),gradV(x2),...)
        # such that we have the total force on each particle immediately.

        # Using the chain rule , we turn the (N,N)−matrix of r−derivatives into
        # the (N,3)−array of derivatives to Cartesian coordinate: the gradient.
        # (Automatically sets diagonal to (0,0,0) = X[ i]−X[ i ])
        u = d/r[:, :, NA]

        # d is an nxnx3 matrix. We want to divide all 3 compontens in x_ijk with r_ij. This is ensured by copying
        # r_ij along the last index. That is, r[:,:,NA]_ijk = r_ij, whereas d_ijk = (xi)_k-(xj)_k

        # u is (N,N,3)−array of unit vectors in direction of X[ i ]−X[ j ],
        # so u is simply the normalized version of
        # displacement = ([x1-x1;x2-x1,...,xN-x1],[x1-x2;x2-x2,...xN-xn],...,[x1-xN;...;xN-xN]

        # Obtain the gradient matrix by multiplying the force magnitudes from T, ie. Tij = -Fij
        # with the corresponding unit vectors between particles i,j

        #

        #obtain the Nx3 gradient matrix = (total gradient of x1; total gradient of x2; ...)
       #T[:,:,NA]*u _ ijk = uijk * (-Fij) -> the k'th component of the negative force between particles i and j

        # Summing over axis 1, the rows, sums all forces on a given particle, so that
        # sum(T[:, :, NA]*u, axis=1)_i = sum of  uijk * (-Fij) over j --> the negative k'th component of the total force on particle i
        # leaving us with gradV = 4*epsilon*sum(T[:, :, NA]*u, axis=1  = (gradV(x1),gradV(x2),...)
        return 4*epsilon*sum(T[:, :, NA]*u, axis=1)
    return gradV


# wrapper functions to generate "flattened" versions of the potential and gradient.
def flatten_function(f):
    return lambda x: f(x.reshape(-1, 3))


def flatten_gradient(f):
    return lambda x: f(x.reshape(-1, 3)).reshape(-1)

# potential and gradient with values for Argon
V = LJ()
gradV = LJgradient()

# Flattened potential and gradient.
flat_V     = flatten_function(V)
flat_gradV = flatten_gradient(gradV)

