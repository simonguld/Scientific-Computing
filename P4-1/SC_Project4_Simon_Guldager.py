import numpy as np
import matplotlib.pyplot as plt

### NB1: The figures in this script have been optimized for full-screen viewing on a 1680x1050 pixel screen.
### IF the figures are not viewed in full-screen with at least this resolution, they will look cluttered.

### NB2: Each differential equation (and the corresponding plotting) takes between 5 and 10 seconds to solve due to the
### high accuracy chosen.


def euler_method_vec(y0,dy,a,b,N):      # y0 is the initial value of y. dy is the differential (vector) eq. to be solved
    h = (b-a) / N                       # Calculate step size
    M = np.size(y0)                     # M is the number of simultaneous equations to be solved
    y = y0                              # Initialize y

    Y = np.empty([M,N])                 # Y is the array in which we store the calculated values of y
    T = np.arange(a,b,h)                # T is the time domain. Note that T_end = b-h
    n = 0                               # Column number of Y
    for t in T:
        Y[:,n] = y                      # Collect value of y
        y = y + h*dy(t,y)               # Update value of y
        n +=1                           # Update column number
    return T,Y
def runge_kutta4(y0,dy,a,b,N):          # y0 is the initial value of y. dy is the differential (vector) eq. to be solved
    h = (b-a) / N                       # Calculate step size
    M = np.size(y0)                     # No. of eq. to be solved
    y = y0                              # Initialize y

    Y = np.empty([M,N])                 # Store calculated values of y in Y
    T = np.arange(a,b,h)                # Time domain. Note that T=b is excluded
    n = 0                               # Column number
    for t in T:
        Y[:, n] = y                     # Store value of y
        n += 1                          # Update column number

        # Calculate k1-k4 characteristic for the 4th order Runge-Kutta method
        k1 = dy(t,y)
        k2 = dy(t + h/2, y + h/2 * k1)
        k3 = dy(t + h/2, y + h/2 * k2)
        k4 = dy(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)       # Update value of y
    return T,Y

def logistic(t,k):          # Logistic function used to test the accuracy of the Euler and Runge-Kutta solvers
    return m / (1 + (m-c)/c * np.exp(-k*t))
def dlogistic (t,y,k):      # Derivate of logistic function
    val = k * y * (1 - y /m)
    return val

def f(t, y, p0, a0, b0, c0, d0, e=0, r=0): #Function of differential equations for simulating HIV-transmission

# f collects the 4 coupled differential equations for the number of infected gay men, bisexual men, straigt women
# and straight men. If mortality is taken into account, so that the total populations of each of these groups
#changes, f collect 8 coupled differential equations.
# If r=0, then the mortality of HIV is ignored, and the population p is constant
# y is a vector variable. If r=0, it is 4-dimensional. If r != 0, so that the population of the 4 groups change,
# then y is 8-dimensional.
# If e=0, the transmission through blood transfusions is ignored

    if r == 0:
        p = p0              # Without mortality, the initial population of the 4 groups
        a = a0
        b = b0
        c = c0
        d = d0

        # Dif. eq. for infected gay men. y[0] = number of infected gay men at time t. p[0] is the total number of gay men
        fx1 = (a[0] * y[0] + a[1] * y[1] + e) * (p[0] - y[0]) - r * y[0]
        # Dif. eq. for infected bisexual men. y[1] = number of infected gay men at time t and p[1] the total no. of bisexual men
        fx2 = (b[0] * y[0] + b[1] * y[1] + b[2] * y[2] + e) * (p[1] - y[1]) - r * y[1]
        # Dif. eq. for infected straight women. y[2] = number of infected gay men at time t and p[2] the total no. of straigt women
        fy = (c[0] * y[1] + c[1] * y[3] + e) * (p[2] - y[2]) - r * y[2]
        # Dif. eq. for infected straight men. y[3] = number of infected straigt men at time t and p[1] the total no. of straigt men
        fz = (d * y[2] + e) * (p[3] - y[3]) - r * y[3]
        # Return derivates
        return np.array([fx1, fx2, fy, fz])

    else:                       # Mortality included
        # y is now an 8-dimensional vector, where y[4], y[5], y[6] and y[7] represent the total population of gay men,
        # bisexual men, straigt women and straight men, respectively
        p = np.ndarray.copy (np.array(y[4::],dtype='float'))

        # the coefficients are given by k/p and thus vary as p does
        a = ( a0 * p0[0] ) / p[0]
        b = ( b0 * p0[1]) / p[1]
        c = (c0 * p0[2]) / p[2]
        d = (d0 * p0[3] ) / p[3]

        # the 4 differential equations for the change in the total populations
        dp = ( np.log(1 -  y[0:4] / p * ( 1 - np.exp(-r))) ) * p

        # the differential equations for the number of infected people in each population, as in the case r=0
        fx1 = (a[0] * y[0] + a[1] * y[1] + e) * (p[0] - y[0]) - r * y[0]
        fx2 = (b[0] * y[0] + b[1] * y[1] + b[2] * y[2] + e) * (p[1] - y[1]) - r * y[1]
        fy = (c[0] * y[1] + c[1] * y[3] + e) * (p[2] - y[2]) - r * y[2]
        fz = (d * y[2] + e) * (p[3] - y[3]) - r * y[3]

        return np.r_['0,1',fx1, fx2, fy, fz,dp]     #Return derivates

def plotter(i,j,k,perturb,method,y,dy,y0,a,b,N,euler=True):
    # This plotter is written to test the Euler and RK4 solvers on a test function
    # It plots the approximation of the method to the function, as well as the approximations of two perturbed inital
    #conditions to test the stability under such perturbations.
    # It also plots the number of significant digits, by comparing the method's value to the exact solution

    # i,j are the index of the position of the graph in the subplot
    # k is a constant parameter and perturb is the value of the perturbation
    # method is either the Euler or RK4 method, y and dy the exact function and its differential
    # y0 is the inital value. a,b and N are the endpoints and number of steps.


    # STEP 1: Solve and plot for exact initial condition
    T, Y = method(y0, lambda t, y: dy(t, y, k), a, b, N)            #Solve differential equation and collect values in Y

    rel_dif_eu = np.abs((y(T, k) - Y[0, :]) / np.abs(y(T, k1)))     # Find relative error for each t in T
    sig_dig_eu = - np.log10(rel_dif_eu)                             # Find no. of sig. digits for each t in T

    # Plot sig. digits and its average value
    axs[i, j].plot(T, sig_dig_eu, 'm-', label=f'y(0) = {c}')
    axs[i, j].plot(T, np.ones(N) * np.average(sig_dig_eu), 'm--')
    # Plot exact function and the approximation
    axs2[i, j].plot(T, Y[0, :], 'm-', label=f'y0 = {c}')
    axs2[i, j].plot(T, logistic(T, k), '-', label=f'Exact X')

    # STEP 2: Solve and plot for perturbed inital condition y0+perturb
    T, Y = method(y0+perturb, lambda t, y: dy(t, y, k), a, b, N)

    rel_dif_eu = np.abs((y(T, k) - Y[0, :]) / np.abs(y(T, k)))
    sig_dig_eu = - np.log10(rel_dif_eu)
    axs[i, j].plot(T, sig_dig_eu, 'r-', label=f'y(0) = {c}+{perturb}')
    axs[i, j].plot(T, np.ones(N) * np.average(sig_dig_eu), 'r--')
    axs2[i, j].plot(T, Y[0, :], 'r-', label=f'y0 = {c}+{perturb}')

    # STEP 3: Solve and plot for perturbed inital condition y0-perturb
    T, Y = method(y0 - perturb, lambda t, y: dy(t, y, k), a, b, N)

    rel_dif_eu = np.abs((y(T, k) - Y[0, :]) / np.abs(y(T, k)))
    sig_dig_eu = - np.log10(rel_dif_eu)
    axs[i, j].plot(T, sig_dig_eu, 'g-', label=f'y(0) = {c}-{perturb}')
    axs[i, j].plot(T, np.ones(N) * np.average(sig_dig_eu), 'g--')
    axs2[i, j].plot(T, Y[0, :], 'g-', label=f'y0 = {c}-{perturb}')


    # Place labels on appropriate subplots
    if i == 1 and j == 0:
        axs2[i, j].set_xlabel('Time', fontweight='bold',fontsize='16')
        axs[i, j].set_xlabel('Time', fontweight='bold',fontsize='16')
        axs[i, j].set_ylabel('Significant Digits', fontweight='bold',fontsize='16')
    elif i == 1 and j ==1:
        axs2[i, j].set_xlabel('Time', fontweight='bold',fontsize='16')
        axs[i, j].set_xlabel('Time', fontweight='bold',fontsize='16')
    elif i == 0 and j == 0:
        axs[i, j].set_ylabel('Significant Digits', fontweight='bold',fontsize='16')

    # Place titles and legends suitable for the Euler method
    if euler:
        axs[i, j].set_title(f'k = {k}', y=0.95, x=0.15, pad=-16, fontweight='bold', fontsize='16')
        axs2[i, j].set_title(f'k = {k}', y=0.95, x=0.15, pad=-16, fontweight='bold', fontsize='16')

        axs[i, j].legend(loc='upper right', ncol=1, fontsize='14')
        axs2[i, j].legend(loc='lower right', ncol=1, fontsize='14')

    # Place titles and legends suitable for the RK4 method
    else:
        axs[i, j].set_title(f'k = {k}', y=0.8, x=0.15, pad=-16, fontweight='bold', fontsize='16')
        axs2[i, j].set_title(f'k = {k}', y=0.95, x=0.15, pad=-16, fontweight='bold', fontsize='16')

        axs[i, j].legend(loc='center right', ncol=1, fontsize='14')
        axs2[i, j].legend(loc='center right', ncol=1, fontsize='14')

def coupled_plotter(ax,i,j,dy,y0,e=0 ,r=0,av=True, sig = False):
    # This plotter is written to calculate and plot no. of HIV-infected individuals for gay men, bisexual men,
    # straight women and straigt women.
    # If sig = True, the no. of significant digits of the Euler approximation are shown for each time step on a
    # 2nd y-axis (called ax). These values are calculated by comparing to the RK4 results and assuming these to be
    # exact. The time step is chosen small enough (1e-3) the the error on the RK4 method is 1e-15, so the assumption is good
    # If av = True, the av. no. of sig. digits of the Euler approximation are shown.

    # i,j are the indices of the position of the subplot
    # dy the vector differential equation to be solved, y0 the initial condition.


    # Solve system of diff. equations with each method
    T, Ybl = runge_kutta4(y0, lambda t, y: f(t, y, p, a, b, c, d, e, r), 0, t1, N)
    T, Y = euler_method_vec(y0, lambda t, y: f(t, y, p, a, b, c, d, e,r), 0, t1, N)

    # Scale the graphs for gay and bisexual men to make plot more readable
    scaling = 7

    ax1[i,j].set_ylim(0,110)
    # Only include labels on the upper left subplot
    if i == 0 and j == 0:
        ax1[i,j].plot(T, scaling*Ybl[0, :], 'm-', label=f'Infected gay men (Scaled)')
        ax1[i,j].plot(T, scaling*Ybl[1, :], 'b-', label=f'Infected bisexual men (Scaled)')
        ax1[i,j].plot(T, Ybl[2, :], 'g-', label=f'Infected straight women')
        ax1[i,j].plot(T, Ybl[3, :], 'r-', label=f'Infected straight men')
    else:
        ax1[i, j].plot(T, scaling*Ybl[0, :], 'm-')
        ax1[i, j].plot(T, scaling*Ybl[1, :], 'b-')
        ax1[i, j].plot(T, Ybl[2, :], 'g-')
        ax1[i, j].plot(T, Ybl[3, :], 'r-')

    # Title on subplots without blood transfusions and mortality
    if r==0 and e==0:
        ax1[i, j].set_title(f'Case {i}{j}', y=0.95, x=0.25, pad=-16, fontweight='bold',fontsize='16')

    # Title on subplots with a given blood transfusion constant e
    if not e==0:
        ax1[i, j].set_title(f'e = {e}', y=0.95, x=0.25, pad=-16, fontweight='bold',fontsize='16')

    # If mortality is included, then y is an 8-dimensional vector because the total populations also change.
    # Plot the total populations on top of the infected populations:
    if not r==0:
        ax1[i,j].plot(T, scaling*Ybl[4,:], 'm--')
        ax1[i,j].plot(T, scaling*Ybl[5,:], 'b--')
        ax1[i,j].plot(T, Ybl[6,:], 'g--')
        ax1[i,j].plot(T, Ybl[7,:], 'r--')
    # Title of subplot with a given mortality rate = % of infected people dying each year / 100
        ax1[i, j].set_title(f'Mortality rate = {1-np.exp(-r):.2f}', y=0.95, x=0.65, pad=-16, fontweight='bold',fontsize='16')
    #  Only include labels on the upper left subplot
        if i == 0 and j == 0:
            ax1[i, j].plot(T, scaling * Ybl[4, :], 'm--',label=f'Population of gay men (Scaled)')
            ax1[i, j].plot(T, scaling * Ybl[5, :], 'b--',label=f'Population of bisexual men (Scaled)')
            ax1[i, j].plot(T, Ybl[6, :], 'g--',label=f'Population of straight women')
            ax1[i, j].plot(T, Ybl[7, :], 'r--',label=f'Population of straight men')

    # If no mortality rate, include second y-axis
    if r==0:
        ax.set_ylim(0, 10)

    # Plot average no. of sig. digits for euler approximation
    if av:
        rel_mg = np.abs(Ybl - Y) / np.abs(Y+1e-14)
        sig_dig = - np.log10(rel_mg+1e-12)

        ax.plot(T, np.ones(N)*np.average(sig_dig[0]), 'm--')
        ax.plot(T, np.ones(N)*np.average(sig_dig[1]), 'b--')
        ax.plot(T, np.ones(N)*np.average(sig_dig[2]), 'g--')
        ax.plot(T, np.ones(N)*np.average(sig_dig[3]), 'r--')
        if i == 0 and j == 0:
            ax.plot(T, np.ones(N) * np.average(sig_dig[0]), 'm--',label=f'Av. sig. digits of Euler for gay men')
            ax.plot(T, np.ones(N) * np.average(sig_dig[1]), 'b--',label=f'Av. sig. digits of Euler for bisexual men')
            ax.plot(T, np.ones(N) * np.average(sig_dig[2]), 'g--',label=f'Av. sig. digits of Euler straight women')
            ax.plot(T, np.ones(N) * np.average(sig_dig[3]), 'r--',label=f'Av. sig. digits of Euler for straight men')

    # Plot no. of sig. digits for euler approximation
    if sig:
        rel_mg = np.abs(Ybl - Y) / np.abs(Y+1e-14)
        sig_dig = - np.log10(rel_mg+1e-12)
        ax.plot(T, sig_dig[0], 'm--')
        ax.plot(T, sig_dig[1], 'b--')
        ax.plot(T, sig_dig[2], 'g--')
        ax.plot(T, sig_dig[3], 'r--')
        if i == 0 and j == 0:
            ax.plot(T, sig_dig[0], 'm--',label='Sig. digits of Euler for gay men')
            ax.plot(T, sig_dig[1], 'b--',label='Sig. digits of Euler for bisexual men')
            ax.plot(T, sig_dig[2], 'g--',label='Sig. digits of Euler for straight women')
            ax.plot(T, sig_dig[3], 'r--',label='Sig. digits of Euler for straight men')

    # Assign labels to appropriate subplots
    if i == 1 and j == 0:
        ax1[i, j].set_xlabel('Time (years)', fontweight='bold',fontsize='16')
        ax1[i, j].set_ylabel('Infected individuals', fontweight='bold',fontsize='16')
    elif i == 1 and j ==1:
        ax1[i, j].set_xlabel('Time (years)', fontweight='bold',fontsize='16')
        if r==0:
            ax.set_ylabel('Significant digits', fontweight='bold',fontsize='16')
    elif i == 0 and j == 0:
        ax1[i, j].set_ylabel('Infected individuals', fontweight='bold',fontsize='16')
    elif i == 0 and j == 1 and r==0:
        ax.set_ylabel('Significant digits', fontweight='bold',fontsize='16')


### The script is broken down into the following sections:

###Problem A: Find a suitable step size for the Euler method by using a logistic test function - as these closely resemble
# the functions of the actual problem of HIV-transmission. Make sure it is robust for different growth rates and
#  under perturbations in initial value

### Problem B: Repeat for the 4th order Runge-Kutta method

### Problem C: Plot the coupled system of equations for the transmission of HIV (ignoring blood transfusion and mortality)
# for different initial conditions (different no. of infect people at t=0)

### Problem D: Plot the coupled system of equations for the transmission of HIV (ignoring mortality) for different
# for values of the blood transfusion constant e to examine its effect

### PROBLEM E: Plot the coupled system of (now 8) differential equations including blood transfusion and mortality for
# different mortality rates to examine its effect


#PROBLEM A: In this part, we will find a 'ideal' step size by considering the accuracy, speed and stability under
# perturbations and different growth rates for the Euler solution to the logistic differential equation

# Define end point
a=1e-14
b=15

m = 10                      # Maximum value of y
y0 = 1                      # Initial value
c = y0

perturb = 1e-2              # Value of perturbation
N=int(15e4)                 # Ideal step size (reasonably fast, accuracy doesn't approve much beyond this)
h = (b-a)/N                 # step size

# Different growth rates considered
k1 = 0.2
k2 = 0.5
k3 = 0.8
k4 = 1.2


fig,axs = plt.subplots(2,2)         # 2x2 subplot for significant digits of Euler method
fig2,axs2 = plt.subplots(2,2)       # 2x2 subplot for Euler approximations to logistic function

# Solve the differential equations for different growth rates and intial value perturbations and plot
plotter(0,0,k1,perturb,euler_method_vec,logistic,dlogistic,y0,a,b,N)
plotter(0,1,k2,perturb,euler_method_vec,logistic,dlogistic,y0,a,b,N)
plotter(1,0,k3,perturb,euler_method_vec,logistic,dlogistic,y0,a,b,N)
plotter(1,1,k4,perturb,euler_method_vec,logistic,dlogistic,y0,a,b,N)

fig.suptitle('Sig. digits for Euler app. to logistic function', fontsize='20',fontweight='bold')
fig2.suptitle('Euler approximation to logistic function', fontsize='20',fontweight='bold')
plt.show()

# We conclude that step length of e-4 is sufficient to make the routine robust from k=0.3 to k=1.2 to more than 4 sig dig,
# which is all we will need to describe the spread of HIV (accuracy of h). We note that the approximations are stable
# under perturbations (the perturbations are conserved)

#PROBLEM B: We repeat problem A but for the 4th order Runge-Kutta implementation

N=int(15e3)
a=0
b = 15
h = (b-a)/N             # Ideal step size is 1e-3. Its error is about 1e-15
range=np.arange(a,b,h)


fig,axs = plt.subplots(2,2)     #Subplot of signifcant digits
fig2,axs2 = plt.subplots(2,2)   #Subplot of solutions
# Solve differential equations for different growth rates and intial value perturbations and plot
plotter(0,0,k1,perturb,runge_kutta4,logistic,dlogistic,y0,a,b,N,euler=False)
plotter(0,1,k2,perturb,runge_kutta4,logistic,dlogistic,y0,a,b,N,euler=False)
plotter(1,0,k3,perturb,runge_kutta4,logistic,dlogistic,y0,a,b,N,euler=False)
plotter(1,1,k4,perturb,runge_kutta4,logistic,dlogistic,y0,a,b,N,euler=False)

fig.suptitle('Sig. digits for Runge-Kutta app. to logistic function', fontsize='20',fontweight='bold')
fig2.suptitle('Runge-Kutta approximation to logistic function', fontsize='20',fontweight='bold')
plt.show()

#We conclude that a step length of 1e-3 is enough to make the implementation accurate to 15 sig. digits. We will there-
# fore assume the RK4 solution of the coupled HIV equations to be exact, thus allowing us to estimate the accuary of the
# Euler implementation, which is only accurate to about 4 sig. digits for the logistic test function

### Problem C: Ignoring mortality and blood transfusion, we will solve and plot the transmission of AIDS for different
# numbers of intially infected people

# Time scale (in years)
t0 = 0
t1 = 30
N= int(3e4)         # N defined so as to ensure a step size of 1e-3

# In Sub-Saharan Africa between the mid-ninetees and the middle of the 2000's, the annual increase people infected
# with HIV ranged between 20% and 300%.
# Assuming that the increase in the number of infected straight men and women is 25% anually, and that the transmission
# risk of receptive anal sex is about 10 times as big as that of vaginal sex,  we infer that the increase in infected gay
# men should be about 8*25% = 200% (not 250%, as not all sexually active gay men receive anal)
# a 200% annual increase translate to an exponential growth rate of ln ( 1+2 ) = 1.099

k = 1.099            # Growth rate assuming that number of infected gay individuals increases by a factor of 4 yearly
k1 = k
k2 = k1            # Growth rate of transmission between gay and bisexual men
k3 = 0.223         # Growth rate of transmission between bisexual men and women, and women and straight men = ln(1+0.25)

p1 = 5              # Total gay population (about 4.5% of all men)
p2 = p1             # Total population of bisexual men (about 4.5% of all men)
q,r = 20 * p1, 20*p1    # Total population of straight women and men (= 100 each)

a1 = k/p1           # gay-gay transmission constant
a2 = k2/p1           # gay-bisexual transmission constant
b1,b2, b3 = k2/p2, k2/p2, k3/p2  #bi-gay, bi-bi and bi-straight transmission constants
c1,c2 = k3/q,k3/q # bi-men-straight women and straight-women-straight-men transmission constants
d = k3/r           # straight-men-straight-women transmission constant



# Collect constants in vectors
p = np.array([p1,p2,q,r], dtype='float')
a = np.array([a1,a2], dtype = 'float')
b = np.array([b1,b2,b3], dtype = 'float')
c = np.array([c1,c2] , dtype='float')
e= 0.01

fig,ax1=plt.subplots(2,2)
# Construct 2nd y-axis for significant digits
ax00 = ax1[0,0].twinx()
# Case 00: 0.5% of gay men infected
x1 = 0.005 * p1
x2, y, z = 0,0,0
y0 = np.array([x1,x2,y,z],dtype='float')
#Solve and plot
coupled_plotter(ax00,0,0,f,y0,sig=True,av=False)

#CASE 01: 20 % of gay men and 20% of bisexual men infected
x1 = 0.2*p1
x2, y, z = x1,0,0
y0 = np.array([x1,x2,y,z],dtype='float')
ax01 = ax1[0,1].twinx()
coupled_plotter(ax01,0,1,f,y0,sig=True,av=False)

# CASE 10: 5 % are infected in each group
y0 = 0.05 * p
ax10 = ax1[1,0].twinx()
coupled_plotter(ax10,1,0,f,y0,sig=True,av=False)

# CASE 11: 25 % are infected in each group
y0 = 0.25 * p
ax11 = ax1[1,1].twinx()
coupled_plotter(ax11,1,1,f,y0,sig=True,av=False)
plt.plot()

fig.legend(loc = 'upper center', ncol=3, fontsize = '10')
plt.show()


### PROBLEM D: Examining the effect of blood transfusions on transmission rates

fig,ax1=plt.subplots(2,2)

# Initial conditions: 0.5 % of gay men are infected.
x1 = 0.005*p1
x2, y, z = 0,0,0
y0 = np.array([x1,x2,y,z],dtype='float')

# Case 1: Yearly infection rate through blood transfusion is 1/10.000 (1 out of 10000 people each year gets infected)
ax00 = ax1[0,0].twinx()
coupled_plotter(ax00,0,0,f,y0,1e-4)

# Case 2: Yearly infection rate through blood transfusion is 1/1000
ax01 = ax1[0,1].twinx()
coupled_plotter(ax01,0,1,f,y0,1e-3)

# Case 2: Yearly infection rate through blood transfusion is 1/500
ax10 = ax1[1,0].twinx()
coupled_plotter(ax10,1,0,f,y0,5e-3)

# Case 3: # Case 2: Yearly infection rate through blood transfusion is 1/100
ax11 = ax1[1,1].twinx()
coupled_plotter(ax11,1,1,f,y0,1e-2)

fig.legend(loc = 'upper center', ncol=3, fontsize='10')
plt.show()


# PROBLEM E: Examine HIV-transmission when including mortality

# Consider a 40 year interval
t1 = 40
N= int(4e3)                 #Reduce stepsize to 1e-2. This gives an accuracy to about 10 sig. digits

fig,ax1=plt.subplots(2,2)

# Blood transfusion: 1/1000 gets infected each year through blood transfusion
e = 1e-3
# Intial conditions:
x1 = 0.005*p1 # 0.5 % of gay men are infected
x2, y, z = 0,0,0
y0 = np.array([x1,x2,y,z],dtype='float')
p = np.array([p1,p2,q,r], dtype='float')
y0mod = np.r_['0,1',y0,p]

# CASE 1: Yearly mortality rate is 1%
mortality_rate = 0.01
r = - np.log(1-mortality_rate)
coupled_plotter(np.nan,0,0,f,y0mod,e,r,av=False)

# CASE 2: Yearly mortality rate is 5%
mortality_rate = 0.05
r = - np.log(1-mortality_rate)
coupled_plotter(np.nan,0,1,f,y0mod,e,r,av=False)

# CASE 3: Yearly mortality rate is 12.5%
mortality_rate = 0.1254
r = - np.log(1-mortality_rate)
coupled_plotter(np.nan,1,0,f,y0mod,e,r,av=False)

# CASE 4: Yearly mortality rate is 25%
mortality_rate = 0.25
r = - np.log(1-mortality_rate)
coupled_plotter(np.nan,1,1,f,y0mod,e,r,av=False)

fig.legend(loc = 'upper center', ncol=3, fontsize='10' )
plt.show()