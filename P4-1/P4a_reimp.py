import numpy as np
import matplotlib.pyplot as plt


def forward_euler_solver(dy, y0, N, t_start, t_end):
    """

    :param dy:
    :param y0:
    :param h:
    :param t_start:
    :param t_end:
    :return:
    """
    dimension = np.size(y0)
    h = (t_end - t_start) / N
    y_values = np.empty([N, dimension])

    if dimension == 1:
        y = y0
        y_values[0] = y
    else:
        y = y0.astype('float')
        y_values[0] = y.astype('float')

    for k in np.arange(1,N):
        y += h * dy(y,t_start + (k-1) * h)
        y_values[k] = y

    if dimension == 1:
        y_values = y_values.flatten()
    return y_values

def fourth_order_runge_kotta(dy, y0, N, t_start, t_end):

    h = (t_end- t_start) / N
    dimension = np.size(y0)
    y_values = np.empty([N, dimension])

    if dimension == 1:
        y = y0
        y_values[0] = y
    else:
        y = y0.astype('float')
        y_values[0] = y.astype('float')

    for k in np.arange(1, N):
        t = t_start + (k-1) * h
        k1 = dy (y, t)
        k2 = dy(y + h / 2 * k1, t + h / 2)
        k3 = dy(y + h / 2 * k2, t + h / 2)
        k4 = dy(y + h * k3, t + h)

        y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y_values[k] = y

    if dimension == 1:
        return y_values.flatten()
    else:
        return y_values

def backward_euler_solver(dy, y0, N, t_start, t_end, local_error_tol, max_iterations):
    h = (t_end - t_start) / N
    dimension = np.size(y0)
    y_values = np.empty([N, dimension])

    if dimension == 1:
        y_values[0] = y0
    else:
        y_values[0] = y0.astype('float')

    for k in np.arange(1,N):

        error = 1
        iterations = 0
        y_old = y_values[k - 1]
        y = y_values[k - 1]

        while np.linalg.norm(error) > local_error_tol and iterations < max_iterations:
            iterations += 1
            dy_new = dy(y, t_start + k * h)
            y = y_old + h * dy_new
            error = y - y_old - h * dy_new

        y_values[k] = y

    if dimension == 1:
        return y_values.flatten()
    else:
        return y_values






def test_function(t, y0, k):
    return y0 * np.exp(k * t)

def test_dev(y, t, k):
    return k * y

#modelling with the parameters given
if 1:
    r1 = 2
    #r3 = r1 / 10
    r3 = r1 / 5
    r4 = r1 / 10


    a1 = np.log(1 + r1)
    a2 = a1

    b1 = a1
    b2 = a1
    b3 = np.log (1 + r4)

    c1 = np.log (1 + r3)
    c2 = c1
    d1 = b3

    p1 = 5
    p2 = 5
    p3 = 100
    p4 = 100

    I1 = 0.05*p1
    I2 = 0
    I3 = 0
    I4 = 0

    coeff = np.array([a1/p1, a2/p1, b1/p2, b2/p2, b3/p2, c1/p3, c2/p3, d1/p4], dtype='float')
    p0 = np.array([p1, p2, p3, p4], dtype='float')
    I0 = np.array([I1, I2, I3, I4], dtype='float')
    I0 = np.r_['0', I0, p0]
   # print(np.size(I0))
    def dy(y, t, P0, a0, e = 0, r = 0):

        coeff = a0.astype('float')
        if r == 0:
            P = P0.astype('float')
        # with mortality rate, y is an 8 dim vector with y[5::] = P(t)
        else:
            P = y.astype('float')[4::]
            y = y.astype('float')[0:4]

            mortality_rate = 1 - np.exp(- r)
            k = np.log(1 - y / P * mortality_rate)

            coeff[0:2] = ( coeff[0:2] * P0[0] ) / P[0]
            coeff[2:5] = (coeff[2:5] * P0[1]) / P[1]
            coeff[5:7] = (coeff[5:7] * P0[2]) / P[2]
            coeff[7] = (coeff[6] * P0[3]) / P[3]

            dP = k * P

        df1 = ( coeff[0] * y[0] + coeff[1] * y[1] + e)  * (P[0] - y[0]) - r * y[0]
        df2 = ( coeff[2] * y[0] + coeff[3] * y[1] + coeff[4] * y[2] + e) * (P[1] - y[1]) - r * y[1]
        df3 = (coeff[5] * y[1] + coeff[6] * y[3] + e) * (P[2] - y[2]) - r * y[2]
        df4 = (coeff[7] * y[2] + e) * (P[3] - y[3]) - r * y[3]
        df = np.array([df1, df2, df3, df4], dtype='float')

        if r != 0:
            return np.r_['0', df, dP]
        return df

    t_start = 0
    t_end = 40
    N = t_end * 1000
    trange = np.linspace(t_start,t_end,N)

   # inf = dy(I0,0,p0 , coeff)
    #df1 = (coeff[0] * I0[0] + coeff[1] * I0[1]) * (p0[0] - I0[0])
    #print(inf)
    mortality_rate = 0.05
    death_coef = - np.log(1-mortality_rate)
    infected = fourth_order_runge_kotta(lambda y, t: dy(y, t, p0, coeff, r = death_coef), I0, N, t_start, t_end)

    plt.plot(trange, 5 * infected[:,0],'b-',label='gay')
    plt.plot(trange, 5 * infected[:,1], 'g-', label='bi')
    plt.plot(trange,  infected[:,2], 'r-', label='women')
    plt.plot(trange,  infected[:,3], 'k-', label='straight')
    plt.legend()
    plt.grid(True)
    plt.show()










if 0:
    y0 = 1
    k = 0.6
    N  = 1000
    t_start = 0
    t_end = 10

    range = np.linspace(t_start,t_end,N)
    values = test_function(range,y0,k)

    values_app = forward_euler_solver(lambda y,t: test_dev(y,t,k), y0, N, t_start, t_end)
    values_app2 = fourth_order_runge_kotta(lambda y, t: test_dev(y, t, k), y0, N, t_start, t_end)
    values_app3 = backward_euler_solver(lambda y, t: test_dev(y, t, k), y0, N, t_start, t_end, 1e-6, 100)

    global_error = np.abs(values - values_app)
    global_error2 = np.abs(values - values_app2)
    global_error3 = np.abs(values - values_app3)

    # plt.figure()
    plt.plot(range, values_app, 'b--', label='D-Eu')
    plt.plot(range, values_app2, 'r--', label='rk4')
    plt.plot(range, values_app3, 'g--', label='B-Eu')
    plt.legend()
    plt.show()

    plt.plot(range, global_error, 'bo', label = f'global error for Direct Euler app to exp function with k = {k}')
    plt.plot(range, global_error2, 'r-', label=f'global error for rk4 app to exp function with k = {k}')
    plt.plot(range, global_error3, 'g-', label=f'global error for B-euler app to exp function with k = {k}')
    plt.legend()

    plt.show()


if 0:
    A = 2
    x0 = -2
    v0 = 0
    y0 = np.array([x0, v0], dtype='float')
    omega = 1

    N = 25000
    t_start = 0
    t_end = 25

    def dxdv(y, t, omega):
        return np.array([y[1], - omega * y[0]])

    def x(t, amplitude, ang_freq, phase):
        return amplitude * np.cos ( ang_freq * t + phase)

    range = np.linspace(t_start, t_end, N)
    values = x(t =  range, amplitude = A, ang_freq= omega, phase = np.pi)

    values_app = forward_euler_solver(lambda y, t: dxdv(y, t, omega), y0, N, t_start, t_end)[:,0]

    values_app2 = fourth_order_runge_kotta(lambda y, t: dxdv(y, t, omega), y0, N, t_start, t_end)[:,0]



    global_error = np.abs(values - values_app)
    global_error_rk4 = np.abs(values-values_app2)
    plt.plot(range, values, 'r-', label=f'true')
    plt.plot(range, values_app, 'b-', label=f'euler')
    plt.plot(range, values_app2, 'm--', label = 'runge')
    plt.grid('True')
    plt.show()

    plt.plot(range, global_error, 'r-', label=f'global error euler')
    plt.plot(range, global_error_rk4, 'g-', label=f'global error rk4')
    plt.legend()
   # plt.plot(range, rel_error, 'b--', label=f'relative')
    plt.show()
