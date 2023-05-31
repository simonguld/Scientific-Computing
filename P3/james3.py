{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "import matplotlib.pyplot as plt, numpy as np\n",
    "from numpy import array, linspace, zeros_like, sign, log2, ceil, identity, argmin, dot, outer\n",
    "from numpy.linalg import solve, norm\n",
    "from LJhelperfunctions import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def restrict(F,gamma): \n",
    "    return lambda t: F(gamma(t))\n",
    "\n",
    "def restrict_grad(gradF,gamma,dgamma): \n",
    "    return lambda t: sum(gradF(gamma(t))*dgamma(t))\n",
    "\n",
    "V_LJ = LJ(SIGMA,EPSILON)\n",
    "gradV_LJ = LJgradient(SIGMA,EPSILON)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7f9d1a1d82b0>]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Assignment a.1\n",
    "# Two Argon atoms at varying distance.\n",
    "d2 = array([[1,0,0],[0,0,0]])\n",
    "\n",
    "V2 = restrict(V_LJ, lambda t: t*d2)\n",
    "xs = linspace(3,11,1000)\n",
    "vs2 = array([V2(x) for x in xs])\n",
    "\n",
    "plt.plot(xs,zeros_like(vs2))\n",
    "plt.plot(xs,vs2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7f9d1a0e6910>]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Assignment a.2\n",
    "# Four Argon atoms: \n",
    "# Same varying diatomic distance, but now two extra Ar atoms add to the potential.\n",
    "d4  = array([[1,0,0],[0,0,0],[0,0,0],[0,0,0]])\n",
    "XX4 = array([[0,0,0],[0,0,0],[14,0,0],[7,3.2,0]])\n",
    "V4 = restrict(V_LJ, lambda t: XX4 + t*d4)\n",
    "\n",
    "xs = linspace(3,11,1000)\n",
    "vs4 = array([V4(x) for x in xs])\n",
    "\n",
    "plt.plot(xs,zeros_like(vs4))\n",
    "plt.plot(xs,vs4)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assignment b\n",
    "def bisection_root(f,a,b,tolerance=1e-13, max_iterations=100):\n",
    "    fa, fb  = f(a), f(b)    \n",
    "        \n",
    "    for i in range(max_iterations):        \n",
    "        m  = a+(b-a)/2\n",
    "        fm = f(m)       \n",
    "        \n",
    "        if(abs(fm)<tolerance): break\n",
    "\n",
    "        if(sign(fm)==sign(fa)):\n",
    "            a, fa = m, fm            \n",
    "        else:\n",
    "            b, fn = m, fm    \n",
    "            \n",
    "    return (a+b)/2, i + 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sigma calculated by bisection is: 3.401 yields energy -7.438e-14 using 48 function calls.\n"
     ]
    }
   ],
   "source": [
    "sigma0,ncalls0=bisection_root(V2,2,6)\n",
    "print(\"sigma calculated by bisection is: \"\n",
    "    +f\"{sigma0:.13} yields energy {V2(sigma0):.4} using {ncalls0} function calls.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Assignment c\n",
    "#c.1\n",
    "def newton_root(f,df,x0,tolerance=1e-14,max_iterations=20):    \n",
    "    x = x0\n",
    "    for i in range(max_iterations):\n",
    "        fx, dfx = f(x), df(x)\n",
    "        x = x - fx/dfx\n",
    "        \n",
    "        if(abs(fx)<tolerance): break\n",
    "    return x, (i+1)*2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sigma calculated by Newton iteration is: 3.401 yields energy -2.657e-15 using 28 function calls.\n"
     ]
    }
   ],
   "source": [
    "#c.2\n",
    "gradV2 = restrict_grad(gradV_LJ, lambda t: t*d2, lambda t: d2)\n",
    "\n",
    "sigma1,ncalls1 = newton_root(V2,gradV2,2)\n",
    "print(\"sigma calculated by Newton iteration is: \"\n",
    "    +f\"{sigma1:.13} yields energy {V2(sigma1):.4} using {ncalls1} function calls.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Assignment d\n",
    "#d.1\n",
    "def guarded_newton_root(f,df, a,b,tolerance=1e-13, max_iterations=1000):\n",
    "    x = a + (b-a)/2\n",
    "    fa, fx, fb = f(a), f(x), f(b)\n",
    "    \n",
    "    assert(sign(fa) != sign(fb))\n",
    "    \n",
    "    for i in range(max_iterations):\n",
    "        # First we take a bisection step\n",
    "        m  = a + (b-a)/2               \n",
    "        fm = f(m)\n",
    "        \n",
    "        if(sign(fa) == sign(fm)): # Right interval,      a < m <= root <= b; new interval [m;b]\n",
    "            a, fa = m, fm\n",
    "        else:                     # Left interval,       a <= root <= m < b; new interval [a;m]\n",
    "            b, fb = m, fm\n",
    "            \n",
    "        # Next we take a Newton-Rhapson step from our best guess x\n",
    "        x  = x - fx/df(x)          \n",
    "        fx = f(x)\n",
    "\n",
    "        if((x<a) or (x>b)):    # If we didn't land within the new bracket, reset x to m\n",
    "            x, fx = m, fm\n",
    "        else:                   # Otherwise use x to update bracket [a;b]\n",
    "            if((sign(fa) == sign(fx))): # Right interval: [x;b]\n",
    "                a, fa = x, fx\n",
    "\n",
    "            if((sign(fb) == sign(fx))): # Left interval: [a;x]\n",
    "                b, fb = x, fx\n",
    "\n",
    "        if(abs(fx) < tolerance): return x, (i+1)*3+3, True\n",
    "\n",
    "    return x, (i+1)*3+3, False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sigma calculated by guarded Newton iteration is: 3.401 yields energy 0.0 using 30 function calls.\n"
     ]
    }
   ],
   "source": [
    "#c.2\n",
    "sigma2,ncalls2,converged = guarded_newton_root(V2,gradV2,2,6,1e-13)\n",
    "\n",
    "print(\"sigma calculated by guarded Newton iteration is: \"\n",
    "    +f\"{sigma2:.13} yields energy {V2(sigma2):.4} using {ncalls2} function calls.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Assignment e"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "(e.1): Only two components are nonzero, because the force is acting along the x-axis. They are equal and opposite due to Newton's third law.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7f9d1a06d5e0>]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#(e.2):\n",
    "gradvs2 = array([gradV2(x) for x in xs])\n",
    "\n",
    "region=(xs>3.5)&(xs<5)\n",
    "plt.plot(xs[region],zeros_like(xs[region]))\n",
    "plt.plot(xs[region],vs2[region])\n",
    "plt.plot(xs[region],gradvs2[region])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.lines.Line2D at 0x7f9d19fd6970>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#(e.3)\n",
    "gradV4  = restrict_grad(gradV_LJ, lambda t: XX4+t*d4, lambda t: d4)\n",
    "gradvs4 = array([gradV4(x) for x in xs])\n",
    "\n",
    "plt.plot(xs[region],zeros_like(xs[region]))\n",
    "plt.plot(xs[region],vs4[region])\n",
    "plt.plot(xs[region],gradvs4[region])\n",
    "imin=np.argmin(vs4)\n",
    "plt.axvline(x=xs[imin],c='black')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "(e.4): The gradient is not zero in the local minimum of the restricted function, as this is not a local minimum of the full n-D function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assignment f\n",
    "# Data\n",
    "XX0 = array([[4,0,0],[0,0,0],[14,0,0],[7,3.2,0]])\n",
    "d0  = -gradV_LJ(XX0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "#f.1\n",
    "def linesearch(F,X0,d,alpha_max=1,tolerance=1e-12,max_iterations=100):\n",
    "    f = lambda alpha: sum(F(X0+alpha*d)*d)\n",
    "    return bisection_root(f,0,alpha_max,tolerance,max_iterations)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "alpha0, ncalls = linesearch(gradV_LJ,XX0,d0)\n",
    "\n",
    "XX1 = XX0+alpha0*d0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.451707051842277"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "alpha0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "################## WEEK 5 #################\n",
    "\n",
    "# Assignment g.1\n",
    "def golden_section_min(f,a,b,tolerance=1e-3): \n",
    "    tau = (sqrt(5)-1)/2\n",
    "    x0, x1  = b - tau*(b-a), a + tau*(b-a)\n",
    "    f0, f1  = f(x0), f(x1)\n",
    "        \n",
    "    n_steps = int(ceil(log2((b-a)/tolerance)/log2(1/tau)))\n",
    "        \n",
    "    for i in range(n_steps):   \n",
    "        assert((a<x0)&(x0<x1)&(x1<b))\n",
    "        if f0<f1:\n",
    "            x0, x1, b = x1 - tau*(x1-a), x0, x1 # Reduce upper bound\n",
    "            f0, f1    = f(x0), f0\n",
    "        else:\n",
    "            a, x0, x1 = x0, x1,  x0+tau*(b-x0)  # Increase lower bound\n",
    "            f0, f1    = f1, f(x1)\n",
    "       \n",
    "    if f0<f1: # Minimum is in [a;x1]\n",
    "        return a+(x1-a)/2,  n_steps + 2\n",
    "    else:     # Minimum is in [x0;b]\n",
    "        return x0+(b-x0)/2, n_steps + 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "# g.2\n",
    "f0  = lambda t: V_LJ(XX0+t*d0)\n",
    "df0 = lambda t: sum(gradV_LJ(XX0+t*d0)*d0)\n",
    "\n",
    "alpha1, ncalls = golden_section_min(f0,0,1,tolerance=1e-12)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.45170706637994085, 0.451707051842277)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "alpha1,alpha0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# g.3 \n",
    "XX0 = array([[0,0,0],[0,0,0]])\n",
    "d0  = array([[0,0,0],[1,0,0]])\n",
    "\n",
    "dopt, ncalls2 = golden_section_min(f0,1,10,tolerance=1e-12)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3.8174934228446955, 64)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dopt, ncalls2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assignment h\n",
    "\n",
    "# Not part of assignment solution, just used for debugging\n",
    "def linesearch_dummy(f,a,b,n_points=500,debug=False):\n",
    "    ts = linspace(a,b,n_points)\n",
    "    fs = array([f(t) for t in ts])\n",
    "    imin = argmin(fs)\n",
    "    alpha, falpha = ts[imin], fs[imin]\n",
    "    \n",
    "    if debug:\n",
    "        plt.plot(ts,fs)\n",
    "        plt.axvline(x=alpha)\n",
    "        plt.show()\n",
    "        print(f\"alpha = {alpha}, f(X0+alpha*dX) = {falpha}\")\n",
    "        \n",
    "    return alpha, n_points\n",
    "\n",
    "# Simple unified BFGS and inverse BFGS implementation.\n",
    "# See \n",
    "# https://www.nbi.dk/~avery/teaching/scicomp/2020/slides/W5ML2b-BFGS.jpg\n",
    "# for the math.\n",
    "def BFGS(f,gradf,X0,tolerance=1e-6, max_iterations=10000, n_reset=100,\n",
    "           BFGSi=False):\n",
    "    n = len(X0)//3\n",
    "    B = identity(3*n) # Broyden approximation to Hessian\n",
    "        \n",
    "    F0 = gradf(X0)\n",
    "    dX = -0.001*F0\n",
    "    \n",
    "    converged = False\n",
    "    N_calls = 1\n",
    "    for i in range(max_iterations):                        \n",
    "        \n",
    "        # Is the solution good enough?\n",
    "        if(sqrt(dot(F0,F0)) < tolerance): \n",
    "            converged = True\n",
    "            break\n",
    "        \n",
    "        # Reset every n_reset steps\n",
    "        if((i%n_reset)==(n_reset-1)): B = identity(3*n)\n",
    "        \n",
    "        # Step 1: Line search along descent direction\n",
    "        f_gamma = lambda t: f(X0+t*dX)\n",
    "        #alpha, n_calls = linesearch_dummy(f_gamma,-10,10, debug=True)\n",
    "        alpha, n_calls = golden_section_min(f_gamma,0,1, tolerance=0.001)\n",
    "        N_calls += n_calls + 1\n",
    "        \n",
    "        # Step 2: Update gradient function value\n",
    "        X1 = X0 + alpha*dX\n",
    "        F1 = gradf(X1)        \n",
    "\n",
    "        # Step 3: Solve the Secant Equation with the BFGS scheme\n",
    "        dY  = F1-F0\n",
    "        \n",
    "        if BFGSi:\n",
    "            # 1. Inverse BFGS update step\n",
    "            BdY = B@dY\n",
    "            B  += outer(dX,dX)/dot(dX,dX) - outer(BdY,BdY)/dot(dY,BdY)            \n",
    "            \n",
    "            # 2. Compute quasi-Newton step with inverse scheme\n",
    "            dX = -B@F1\n",
    "        else:\n",
    "            # 1. BFGS update step\n",
    "            BdX = B@dX                      \n",
    "            B  += outer(dY,dY)/dot(dY,dY) - outer(BdX, BdX)/dot(dX,BdX)\n",
    "            \n",
    "            # 2. Compute quasi-Newton step with direct scheme\n",
    "            dX  = solve(B,-F1)\n",
    "                        \n",
    "        X0, F0 = X1, F1\n",
    "    \n",
    "    return X1, N_calls, converged\n",
    "        \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "# h.2\n",
    "f = np.load(\"ArStart.npz\")\n",
    "XX0        = f['Xstart2']\n",
    "flat_V     = flatten_function(V_LJ)\n",
    "flat_gradV = flatten_gradient(gradV_LJ)\n",
    "flat_XX0   = XX0.flatten()\n",
    "\n",
    "X_opt_flat,N_calls, converged = myBFGS(flat_V,flat_gradV,flat_XX0, BFGSi=False)\n",
    "X_opt = X_opt_flat.reshape(-1,3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([[0.  , 3.82],\n",
       "        [3.82, 0.  ]]),\n",
       " 55)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.round(distance(X_opt),2), N_calls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimal two-particle distance is 3.8175Å\n"
     ]
    }
   ],
   "source": [
    "r0 = distance(X_opt)[1,0]\n",
    "print(f\"Optimal two-particle distance is {r0:.5}Å\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assignment i/j\n",
    "def count_bonds(X_opt):\n",
    "    D = distance(X_opt)\n",
    "    return sum(abs(D-r0)/r0 <= 0.01)//2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# i.1\n",
    "count_bonds(X_opt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Xstart2: V(X_opt)=-0.997 kJ/mol and 1 bonds, converged after 55 calls\n",
      "Xstart3: V(X_opt)=-2.991 kJ/mol and 3 bonds, converged after 181 calls\n",
      "Xstart4: V(X_opt)=-5.982 kJ/mol and 6 bonds, converged after 1045 calls\n",
      "Xstart5: V(X_opt)=-9.077 kJ/mol and 9 bonds, converged after 4969 calls\n",
      "Xstart6: V(X_opt)=-12.27 kJ/mol and 12 bonds, converged after 2917 calls\n",
      "Xstart7: V(X_opt)=-15.49 kJ/mol and 15 bonds, converged after 2413 calls\n",
      "Xstart8: V(X_opt)=-19.76 kJ/mol and 18 bonds, converged after 6337 calls\n",
      "Xstart9: V(X_opt)=-23.08 kJ/mol and 17 bonds, converged after 4879 calls\n",
      "Xstart20: V(X_opt)=-68.48 kJ/mol and 23 bonds, converged after 18811 calls\n"
     ]
    }
   ],
   "source": [
    "for n in [2,3,4,5,6,7,8,9,20]:\n",
    "    XX0 = f[f'Xstart{n}']\n",
    "    flat_XX0   = XX0.flatten()\n",
    "\n",
    "    X_opt_flat,N_calls, converged = myBFGS(flat_V,flat_gradV,flat_XX0, BFGSi=True)\n",
    "    X_opt = X_opt_flat.reshape(-1,3)\n",
    "    \n",
    "    print(f\"Xstart{n}: V(X_opt)={V_LJ(X_opt):.4} kJ/mol and {count_bonds(X_opt)} bonds, {'converged' if converged else 'failed'} after {N_calls} calls\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
