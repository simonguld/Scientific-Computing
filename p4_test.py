import numpy as np
import matplotlib.pyplot as plt


### NB1: The figures in this script have been optimized for full-screen viewing on a 2560x1440 pixel screen.
### IF the figures are not viewed in full-screen with this resolution, they will look cluttered.

### NB2: Each differential equation (and the corresponding plotting) takes between 5 and 10 seconds to solve due to the
### high accuracy chosen.


def euler_method_vec(y0,dy,a,b,N):
    h = (b-a) / N
    M = np.size(y0)
    y = y0                              #Mx1

    Y = np.empty([M,N])
    T = np.arange(a,b,h)
    n = 0
    for t in T:
        Y[:,n] = y
        y = y + h*dy(t,y)
        n +=1
    return T,Y

def runge_kutta4(y0,dy,a,b,N):
    h = (b-a) / N
    M = np.size(y0)
    y = y0                            #Mx1

    Y = np.empty([M,N])
    T = np.arange(a,b,h)
    n = 0
    for t in T:
        Y[:, n] = y
        n += 1
        k1 = dy(t,y)
        k2 = dy(t + h/2, y + h/2 * k1)
        k3 = dy(t + h/2, y + h/2 * k2)
        k4 = dy(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return T,Y

def dlogistic (t,y,k):
    val = k * y * (1 - y /m)
    #val = k * y
    return val

def logistic(t,k):
    return m / (1 + (m-c)/c * np.exp(-k*t))
    #return np.exp(k*t)

def f(t, y, p0, a0, b0, c0, d0, e=0, r=0):

    if r == 0:
        p = p0

        a = a0
        b = b0
        c = c0
        d = d0

        fx1 = (a[0] * y[0] + a[1] * y[1] + e) * (p[0] - y[0]) - r * y[0]
        fx2 = (b[0] * y[0] + b[1] * y[1] + b[2] * y[2] + e) * (p[1] - y[1]) - r * y[1]
        fy = (c[0] * y[1] + c[1] * y[3] + e) * (p[2] - y[2]) - r * y[2]
        fz = (d * y[2] + e) * (p[3] - y[3]) - r * y[3]
        return np.array([fx1, fx2, fy, fz])

    else:
        p = np.copy ( y[4::])


        a = ( a0 * p0[0] ) / p[0]
        b = ( b0 * p0[1]) / p[1]
        c = (c0 * p0[2]) / p[2]
        d = (d0 * p0[3] ) / p[3]

        dp = ( np.log(1 -  y[0:4] / p * ( 1 - np.exp(-r))) ) * p

        fx1 = (a[0] * y[0] + a[1] * y[1] + e) * (p[0] - y[0]) - r * y[0]
        fx2 = (b[0] * y[0] + b[1] * y[1] + b[2] * y[2] + e) * (p[1] - y[1]) - r * y[1]
        fy = (c[0] * y[1] + c[1] * y[3] + e) * (p[2] - y[2]) - r * y[2]
        fz = (d * y[2] + e) * (p[3] - y[3]) - r * y[3]
        return np.r_['0,1',fx1, fx2, fy, fz,dp]

def plotter(i,j,k,perturb,method,y,dy,y0,a,b,N,euler=True):
    T, Y = method(y0, lambda t, y: dy(t, y, k), a, b, N)

    rel_dif_eu = np.abs((y(T, k) - Y[0, :]) / np.abs(y(T, k1)))
    sig_dig_eu = - np.log10(rel_dif_eu)
    axs[i, j].plot(T, sig_dig_eu, 'm-', label=f'y(0) = {c}')
    axs[i, j].plot(T, np.ones(N) * np.average(sig_dig_eu), 'm--')
    axs2[i, j].plot(T, Y[0, :], 'm-', label=f'y0 = {c}')
    axs2[i, j].plot(T, logistic(T, k), '-', label=f'Exact X')


    T, Y = method(y0+perturb, lambda t, y: dy(t, y, k), a, b, N)

    rel_dif_eu = np.abs((y(T, k) - Y[0, :]) / np.abs(y(T, k)))
    sig_dig_eu = - np.log10(rel_dif_eu)
    axs[i, j].plot(T, sig_dig_eu, 'r-', label=f'y(0) = {c}+{perturb}')
    axs[i, j].plot(T, np.ones(N) * np.average(sig_dig_eu), 'r--')
    axs2[i, j].plot(T, Y[0, :], 'r-', label=f'y0 = {c}+{perturb}')

    T, Y = method(y0 - perturb, lambda t, y: dy(t, y, k), a, b, N)

    rel_dif_eu = np.abs((y(T, k) - Y[0, :]) / np.abs(y(T, k)))
    sig_dig_eu = - np.log10(rel_dif_eu)
    axs[i, j].plot(T, sig_dig_eu, 'g-', label=f'y(0) = {c}-{perturb}')
    axs[i, j].plot(T, np.ones(N) * np.average(sig_dig_eu), 'g--')
    axs2[i, j].plot(T, Y[0, :], 'g-', label=f'y0 = {c}-{perturb}')


    #axs[i, j].set_title(f'Sig. digits for k = {k}')
    #axs2[i, j].set_title(f'Approximations for k = {k}')


    if i == 1 and j == 0:
        axs2[i, j].set_xlabel('Time', fontweight='bold',fontsize='16')
        axs[i, j].set_xlabel('Time', fontweight='bold',fontsize='16')
        axs[i, j].set_ylabel('Significant Digits', fontweight='bold',fontsize='16')
    elif i == 1 and j ==1:
        axs2[i, j].set_xlabel('Time', fontweight='bold',fontsize='16')
        axs[i, j].set_xlabel('Time', fontweight='bold',fontsize='16')
    elif i == 0 and j == 0:
        axs[i, j].set_ylabel('Significant Digits', fontweight='bold',fontsize='16')

    if euler:
        axs[i, j].set_title(f'k = {k}', y=0.95, x=0.15, pad=-16, fontweight='bold', fontsize='16')
        axs2[i, j].set_title(f'k = {k}', y=0.95, x=0.15, pad=-16, fontweight='bold', fontsize='16')

        axs[i, j].legend(loc='upper right', ncol=1, fontsize='14')
        axs2[i, j].legend(loc='lower right', ncol=1, fontsize='14')

    else:
        axs[i, j].set_title(f'k = {k}', y=0.8, x=0.15, pad=-16, fontweight='bold', fontsize='16')
        axs2[i, j].set_title(f'k = {k}', y=0.95, x=0.15, pad=-16, fontweight='bold', fontsize='16')

        axs[i, j].legend(loc='center right', ncol=1, fontsize='14')
        axs2[i, j].legend(loc='center right', ncol=1, fontsize='14')
        #axs2[i, j].legend(y=0.2, x=0.9, pad=-16, ncol=1, fontsize='14')

def coupled_plotter(ax,i,j,dy,y0,e=0 ,r=0,av=True, sig = False):
    T, Ybl = runge_kutta4(y0, lambda t, y: f(t, y, p, a, b, c, d, e, r), 0, t1, N)
    T, Y = euler_method_vec(y0, lambda t, y: f(t, y, p, a, b, c, d, e,r), 0, t1, N)


    scaling = 7
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

    if r==0 and e==0:
        ax1[i, j].set_title(f'Case {i}{j}', y=0.95, x=0.25, pad=-16, fontweight='bold',fontsize='16')


    if not e==0:
        ax1[i, j].set_title(f'e = {e}', y=0.95, x=0.25, pad=-16, fontweight='bold',fontsize='16')
    if not r==0:
        ax1[i,j].plot(T, scaling*Ybl[4,:], 'm--')
        ax1[i,j].plot(T, scaling*Ybl[5,:], 'b--')
        ax1[i,j].plot(T, Ybl[6,:], 'g--')
        ax1[i,j].plot(T, Ybl[7,:], 'r--')

        ax1[i, j].set_title(f'Mortality rate = {1-np.exp(-r):.2f}', y=0.95, x=0.65, pad=-16, fontweight='bold',fontsize='16')
        if i == 0 and j == 0:
            ax1[i, j].plot(T, scaling * Ybl[4, :], 'm--',label=f'Population of gay men (Scaled)')
            ax1[i, j].plot(T, scaling * Ybl[5, :], 'b--',label=f'Population of bisexual men (Scaled)')
            ax1[i, j].plot(T, Ybl[6, :], 'g--',label=f'Population of straight women')
            ax1[i, j].plot(T, Ybl[7, :], 'r--',label=f'Population of straight men')


    #plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
    #plt.rcParams['axes.titlepad'] = -14  # pad is in points...


    ax1[i,j].set_ylim(0,110)
    if r==0:
        ax.set_ylim(0, 10)

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



#problem a: Find a suitable step size for euler - robust under changes i r and starting value. Use logistic test
# function. Way speed with number of sig digits of figure out how many are feasible
#problem b: Repeat for Runge-Kutta. Decide the accuracy you want and find step size accordingly.
## OBS: You could choose a small step size to obtain superior accurary, thus allowing you to treat the ode4 case as
# the right one.
#problem c: Plot coupled system without blood and death. Experiment with constant terms (a1, and inital sick population).
# Scale populatoin and possibly time. Consider tweaking a1 to obtain a less crazy growth. Compare Eu and ode4
#Problem d: Include blood transfusions. Plot for different values of and examine effects. Compare Eu and ode4
#Problem e: Include death rates. Consider including population decay. Try tweaking r to see what happens. Compare Eu and ode4.

# PROBLEM A:
# Find app. step size for euler using a logistic test function.

a=1e-14
b=20
m = 10
y0 = 1
c = 1
perturb = 1e-2
N=int(2e5)      # SET BACK
h = (b-a)/N
range=np.arange(a,b,h)
k1 = 0.1
k2 = 0.3
k3 = 0.5
k4 = 0.7



fig,axs = plt.subplots(2,2)
fig2,axs2 = plt.subplots(2,2)
plotter(0,0,k1,perturb,euler_method_vec,logistic,dlogistic,y0,a,b,N)
plotter(0,1,k2,perturb,euler_method_vec,logistic,dlogistic,y0,a,b,N)
plotter(1,0,k3,perturb,euler_method_vec,logistic,dlogistic,y0,a,b,N)
plotter(1,1,k4,perturb,euler_method_vec,logistic,dlogistic,y0,a,b,N)


fig.suptitle('Sig. digits for Euler app. to logistic function', fontsize='20',fontweight='bold')
fig2.suptitle('Euler approximation to logistic function', fontsize='20',fontweight='bold')

#fig.legend(loc = 'upper center', ncol=3, fontsize = '14')
plt.show()



plt.show()

# We conclude that step length of e-4 is sufficient to make the routine robust from k=0.1 to k=0.7 to more than 4.5 sig dig, which is all we will
# need to describe the spread of HIV (accuracy of h). Stable under perturbations (original perturbation is conserved)

#PROBLEM B: Find the optimal step size for Runge- Kutta

N=int(2e5)
a=0
b = 20
h = (b-a)/N
range=np.arange(a,b,h)

fig,axs = plt.subplots(2,2)
fig2,axs2 = plt.subplots(2,2)
plotter(0,0,k1,perturb,runge_kutta4,logistic,dlogistic,y0,a,b,N,euler=False)
plotter(0,1,k2,perturb,runge_kutta4,logistic,dlogistic,y0,a,b,N,euler=False)
plotter(1,0,k3,perturb,runge_kutta4,logistic,dlogistic,y0,a,b,N,euler=False)
plotter(1,1,k4,perturb,runge_kutta4,logistic,dlogistic,y0,a,b,N,euler=False)

fig.suptitle('Sig. digits for Runge-Kutta app. to logistic function', fontsize='20',fontweight='bold')
fig2.suptitle('Runge-Kutta approximation to logistic function', fontsize='20',fontweight='bold')
#fig.legend(loc = 'upper center', ncol=3, fontsize = '14')
plt.show()

#We conclude that a step length of 1e-3 is enough to make accurate to e-15. Stable under perturbations

#problem c: Plot coupled system without blood and death. Experiment with constant terms (a1, and inital sick population).
# Scale population and possibly time. Consider tweaking a1 to obtain a less crazy growth. Compare Eu and ode4

t0 = 0
t1 = 20
N= int(2e4)         #CHANGE PRECISION
k = 0.3

p1 = 5
p2 = p1
q,r = 20 * p1, 20*p1

a1 = k/p1
a2 = a1/2
b1,b2, b3 = a1/2, a1/10, a1/10
c1,c2 = a1/10,a1/10
d = a1/10

x1 = p1/500
x2, y, z = 0,0,0

y0 = np.array([x1,x2,y,z],dtype='float')
p = np.array([p1,p2,q,r], dtype='float')
a = np.array([a1,a2], dtype = 'float')
b = np.array([b1,b2,b3], dtype = 'float')
c = np.array([c1,c2] , dtype='float')
e= 0.01





#WITHOUT BLOOD NOR DEATH
fig,ax1=plt.subplots(2,2)

x1 = 0.005*p1 # 0.5 % af bøsser infected
x2, y, z = 0,0,0

y0 = np.array([x1,x2,y,z],dtype='float')
ax00 = ax1[0,0].twinx()
#ax00.axis('off')
coupled_plotter(ax00,0,0,f,y0,sig=True,av=False)

#20% af bøsser og 20% af bi's infected
x1 = 0.2*p1
x2, y, z = x1,0,0
y0 = np.array([x1,x2,y,z],dtype='float')
ax01 = ax1[0,1].twinx()
coupled_plotter(ax01,0,1,f,y0,sig=True,av=False)

# 5% af alle infected
y0 = p / 20
ax10 = ax1[1,0].twinx()
coupled_plotter(ax10,1,0,f,y0,sig=True,av=False)
#ax10.axis('off')

# 25% af alle infected
y0 = 0.25 * p
ax11 = ax1[1,1].twinx()
coupled_plotter(ax11,1,1,f,y0,sig=True,av=False)
plt.plot()

fig.legend(loc = 'upper center', ncol=3, fontsize = '14')
plt.show()



# WITH BLOOD
fig,ax1=plt.subplots(2,2)

x1 = 0.005*p1 # 0.5% af bøsser infected. Alle andre raske.
x2, y, z = 0,0,0
y0 = np.array([x1,x2,y,z],dtype='float')


ax00 = ax1[0,0].twinx()

coupled_plotter(ax00,0,0,f,y0,1e-4)


ax01 = ax1[0,1].twinx()
coupled_plotter(ax01,0,1,f,y0,1e-3)

ax10 = ax1[1,0].twinx()
coupled_plotter(ax10,1,0,f,y0,1e-2)

ax11 = ax1[1,1].twinx()
coupled_plotter(ax11,1,1,f,y0,1e-1)


fig.legend(loc = 'upper center', ncol=3, fontsize='14')
plt.show()


# WITH BLOOD, but just 0.1% of population getting infected yearly.
# WITH DEATH

t1 = 40
N= int(4e3)                 #NB changed precision

fig,ax1=plt.subplots(2,2)
e = 1e-3

x1 = 0.005*p1 # 5 % af bøsser infected. 5% af bi
x2, y, z = 0,0,0
y0 = np.array([x1,x2,y,z],dtype='float')

p = np.array([p1,p2,q,r], dtype='float')
y0mod = np.r_['0,1',y0,p]



mortality_rate = 0.01
r = - np.log(1-mortality_rate)
coupled_plotter(np.nan,0,0,f,y0mod,e,r,av=False)

mortality_rate = 0.05
r = - np.log(1-mortality_rate)

coupled_plotter(np.nan,0,1,f,y0mod,e,r,av=False)


mortality_rate = 0.1254
r = - np.log(1-mortality_rate)

coupled_plotter(np.nan,1,0,f,y0mod,e,r,av=False)


mortality_rate = 0.25
r = - np.log(1-mortality_rate)

coupled_plotter(np.nan,1,1,f,y0mod,e,r,av=False)


fig.legend(loc = 'upper center', ncol=3, fontsize='14' )
plt.show()





