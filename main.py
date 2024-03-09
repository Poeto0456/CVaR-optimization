# -*- coding: utf-8 -*-
"""
Created on Fri Nov  21 17:26:27 2023

@author: jaimungal
"""

import numpy as np
from collections import deque
import copy
import params

from tqdm import tqdm

seed = params.params['seed']
np.random.seed(seed)

N_pools = params.N_pools

Rx0 = params.params['Rx0']
Ry0 = params.params['Ry0'] 
phi = params.params['phi']

batch_size = params.params['batch_size']
T          = params.params['T']
kappa      = params.params['kappa']
p          = params.params['p']
sigma      = params.params['sigma']

alpha = params.params['alpha']
q = params.params['q']
x_0 = params.params['x_0']
zeta = params.params['zeta']

class amm():

    def __init__(self, Rx, Ry,  phi):
        """
        instantiate the class

        Parameters
        ----------
        Rx : array (K,)
            initial reservese of token-X in each pool
        Ry : array (K,)
            initial reservese of token-Y in each pool
        phi : array (K,)
            the pool fee

        Returns
        -------
        None.

        """
        
        assert (len(Rx) == len(Ry)) & (len(Ry)==len(phi)), "length of Rx, Ry, and phi must be the same."

        self.Rx = 1*Rx
        self.Ry = 1*Ry
        self.phi = 1*phi
        self.N = len(self.Rx)

        # number of LP tokens for each pool
        self.L = np.sqrt(self.Rx*self.Ry)

        # the trader begins with no LP tokens
        self.l = np.zeros(len(self.L))

    def swap_x_to_y(self, x, quote=False):
        """
        swap token-X for token-Y across all pools simulataneously

        Parameters
        ----------
        x : array (K,)
            the amount of token-X to swap in each AMM pool.
        quote: bool, optional
            deafult is False.
            If False, then pool states are updated.
            If True, pool states are not updated.

        Returns
        -------
        y : array (K,)
            the amount of token-Y you receive from each pool.

        """

        y = (x * (1 - self.phi) * self.Ry) / (self.Rx + (1 - self.phi) * x)

        if not quote:
            self.Rx += x
            self.Ry -= y

        return y

    def swap_y_to_x(self, y, quote=False):
        """
        swap token-Y for token-X across all pools simulataneously

        Parameters
        ----------
        y : array (K,)
            the amount of token-Y to swap in each AMM pool.
        quote: bool, optional
            deafult is False.
            If False, then pool states are updated.
            If True, pool states are not updated.

        Returns
        -------
        x : array (K,)
            the amount of token-X you receive from each pool.

        """
        
        x = (y * (1 - self.phi) * self.Rx) / (self.Ry + (1 - self.phi) * y)

        if not quote:
            self.Rx -= x
            self.Ry += y

        return x

    def mint(self, x, y):
        """
        mint LP tokens across all pools

        Parameters
        ----------
        x : array (K,)
            amount of token-X submitted to each pool.
        y : array (K,)
            amount of token-Y submitted to each pool.

        Returns
        -------
        l : array (K,)
            The amount of LP tokens you receive from each pool.

        """

        for k in range(len(self.Rx)):
            assert np.abs(((x[k]/y[k])-self.Rx[k]/self.Ry[k])) < 1e-9, "pool " + str(k) + " has incorrect submission of tokens"

        l = x / self.Rx * self.L
        self.Rx += x
        self.Ry += y
        self.L += l
        self.l += l

        return l

    def swap_and_mint(self, x):
        """
        a method that determines the correct amount of y for each x within the corresponding pool
        to swap and then mint tokens with the reamaing x and the y you received

        Parameters
        ----------
        x : array (K,)
            amount of token-X you have for each pool.

        Returns
        -------
        l : array (K,)
            The amount of LP tokens you receive from each pool.

        """
        
        theta = (1 + (((2 - self.phi) * self.Rx) / (2 * (1 - self.phi) * x)) * (1 - np.sqrt(1 + 4 * (x / self.Rx) * (1 - self.phi) / (2 - self.phi)**2)))
        new_x = (1 - theta) * x

        y = (new_x * (1 - self.phi) * self.Ry) / (self.Rx + (1- self.phi) * new_x)
        self.Rx += new_x
        self.Ry -= y

        x = theta * x
        l = x / self.Rx * self.L
        self.Rx += x
        self.Ry += y
        self.L += l
        self.l += l

        return l
    
    def burn_and_swap(self, l):
        """
        a method that burns your LP tokens, then swaps y to x and returns only x

        Parameters
        ----------
        l : array (K,)
            amount of LP tokens you have for each pool.

        Returns
        -------
        x : array (K,)
            The amount of token-x you receive at the end.

        """
        x = l / self.L * self.Rx
        y = l / self.L * self.Ry
        sum_x = np.sum(x)
        self.Rx -= x
        self.Ry -= y
        self.l -= l
        self.L -= l


        total_y = np.sum(y)
        bestPrice = 0

        for i in range(len(self.Rx)):
            x[i] = (total_y * (1 - self.phi[i]) * self.Rx[i]) / (self.Ry[i] + (1 - self.phi[i]) * total_y)
            if x[i] > bestPrice:
                bestPrice = x[i]

        total_x = bestPrice + sum_x

        return total_x

    def burn(self, l):
        """
        burn LP tokens across all pools

        Parameters
        ----------
        l : array (K,)
            amount of LP tokens to burn

        Returns
        -------
        x : array (K,)
            The amount of token-X received across
        y : array (K,)
            The amount of token-Y received across

        """

        for k in range(len(self.L)):
            assert l[k] <= self.l[k], "you have insufficient LP tokens"

        x = l / self.L * self.Rx
        y = l / self.L * self.Ry
        self.Rx -= x
        self.Ry -= y
        self.L -= l
        self.l -= l
        
        return x, y

    def simulate(self, kappa, p, sigma, T=1, batch_size=256):
        """
        Simulate trajectories of all AMM pools simultanesouly.

        Parameters
        ----------
        kappa : array (K+1,)
            rate of arrival of swap events X->Y and Y->X.
            kappa[0,:] is for a common event across all pools
        p : array (K+1,2)
            probability of swap X to Y event conditional on an event arriving.
            p[0,:] is for a common event across all pools
        sigma : array (K+1,2)
            standard deviation of log volume of swap events.
            sigma[0,:] is for a common event across all pools
        T : float, optional: default is 1.
            The amount of (calendar) time to simulate over
        batch_size : int, optional, default is 256.
            the number of paths to generate.

        Returns
        -------
        pools : deque, len=batch_size
            Each element of the list is the pool state at the end of the simulation for that scenario
        Rx_t : deque, len= batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the reserves in token-X for all AMM pools after each transaction.
        Ry_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the reserves in token-Y for all AMM pools after each transaction.
        v_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the volumes of the transaction sent to the various AMM pools -- the transaction is
            either a swap X for Y or swap Y for X for a single pool, or across all pools at once
        event_type_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the event type. event_type=0 if it is a swap sent to all pools simultaneously,
            otherwise, the swap was sent to pool event_type
        event_direction_t : deque, len=batch_size
            Each element of the list contains a sequence of arrays.
            Each array shows the directi of the swap.
            event_direction=0 if swap X -> Y
            event_direction=1 if swap Y -> X

        """

        # used for generating Poisson random variables for all events
        sum_kappa = np.sum(kappa)

        # used for thinning the Poisson process
        pi = kappa/sum_kappa

        # store the list of reservese generated by the simulation
        def make_list(batch_size):
          x = deque(maxlen=batch_size)
          x = [None] * batch_size
          return x

        Rx_t = make_list(batch_size)
        Ry_t = make_list(batch_size)       
        v_t = make_list(batch_size)
        event_type_t = make_list(batch_size)
        event_direction_t = make_list(batch_size)
        pools = make_list(batch_size)

        for k in tqdm(range(batch_size)):

            N = np.random.poisson(lam = sum_kappa*T)

            Rx = np.zeros((N,len(self.Rx)))
            Ry = np.zeros((N,len(self.Rx)))
            v = np.zeros((N,len(self.Rx)))
            event_type = np.zeros(N, int)
            event_direction = np.zeros(N, int)

            pools[k] = copy.deepcopy(self)

            for j in range(N):

                # generate the type of event associated with each event
                event_type[j] = np.random.choice(len(kappa), p=pi)

                # the direction of the swap 0 = x-> y, 1 = y -> x
                event_direction[j] = int(np.random.rand()< p[event_type[j]])

                if event_direction[j] == 0:
                    mu = np.zeros(len(pools[k].Rx)) # deposit X and get Y
                else:
                    mu = np.log(pools[k].Ry/pools[k].Rx) # deposit Y and get X

                if event_type[j] == 0:
                    # there is a swap across all venues
                    v[j,:] = np.exp((mu-0.5*sigma[0]**2) + sigma[0]*np.random.randn() )

                else:
                    # there is a swap only on a specific venue
                    v[j,:] = np.zeros(len(pools[k].Rx))
                    mu = mu[event_type[j]-1]
                    v[j,event_type[j]-1] = np.exp((mu-0.5*sigma[event_type[j]]**2) \
                                                   + sigma[event_type[j]]*np.random.randn() )

                if event_direction[j] == 0:
                    pools[k].swap_x_to_y(v[j,:]) # submit X and get Y
                else:
                    pools[k].swap_y_to_x(v[j,:]) # submit Y and get X

                Rx[j,:] = 1*pools[k].Rx
                Ry[j,:] = 1*pools[k].Ry

            Rx_t[k] = Rx
            Ry_t[k] = Ry
            v_t[k] = v
            event_type_t[k] = event_type
            event_direction_t[k] = event_direction

        return pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t
    
    def total_coins_of_x(self, l0, batch_size, kappa, p, sigma, T):
        global seed
        np.random.seed(seed)
        end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t =\
        self.simulate( kappa = kappa, p = p, sigma = sigma, T = T, batch_size = batch_size)
        x_T = np.zeros(batch_size)
        for k in range(batch_size):
            x_T[k] = np.sum(end_pools[k].burn_and_swap(l0))
        return x_T

    def estimate_gradient(self,theta,initial_wealth):
        global zeta
        global alpha
        global seed
        np.random.seed(seed)
        global batch_size
        global T
        global kappa
        global p
        global sigma

        theta_array = np.concatenate(([theta], np.random.dirichlet(np.ones(6), 10)))
        P = self.P(theta_array[0], 1, initial_wealth)
        base_function = self.estimate_cvar(theta,batch_size,kappa,p,sigma,T) + P

        estimate = []
        for i in range(1, len(theta_array)):
            cvar = self.estimate_cvar(theta_array[i],batch_size,kappa,p,sigma,T)
            function = cvar + self.P(theta_array[i], 1, initial_wealth)
            estimate.append((function - base_function)/((theta_array[i] - theta_array[0])))

        return np.mean(estimate,axis=0)

    def project_to_simplex(self, v, z = 1):
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.nonzero(u > cssv)[0][-1]
        theta = np.maximum(v - cssv[ind] / (ind+1), 0)
        return theta / np.sum(theta)
        
    def P(self, theta, r, initial_wealth):
        global seed
        np.random.seed(seed)
        global batch_size
        global T
        global kappa
        global p
        global sigma
        global zeta
        global q
        POOLS = copy.deepcopy(self)

        l    = POOLS.swap_and_mint(theta * initial_wealth)
        x_T = POOLS.total_coins_of_x(l, batch_size, kappa, p, sigma, T)
        log_ret  = np.log(x_T) - np.log(initial_wealth)

        P = (log_ret < zeta).mean()

        return r * max(0, P - (1 - q))**2
    
    def SGD(self):
        global seed
        np.random.seed(seed)
        global batch_size
        global T
        global kappa
        global p
        global sigma
        global x_0

        learning_rate = 0.08
        theta_list = [np.array([0.11034734, 0.20812356, 0.21412099, 0.23870377, 0.12366269, 0.10504165])]
        cvar_estimates = []

        for i in range(1, 100):
            g_t = self.estimate_gradient(theta_list[i-1], x_0)
            theta_new = theta_list[i-1] - learning_rate*g_t 
            theta_proj = self.project_to_simplex(theta_new)        
            theta_list.append(theta_proj)
            cvar  = self.estimate_cvar(theta_proj, batch_size, kappa, p, sigma, T)
            print(theta_proj)
            print(cvar)
            cvar_estimates.append(cvar)
            
        
        theta_array_ = np.array(theta_list[1:])

        return np.mean(theta_array_, axis = 0), cvar_estimates
    
    def estimate_cvar(self, theta, batch_size, kappa, p, sigma, T):
        global x_0
        global alpha

        POOLS = copy.deepcopy(self)
        l = POOLS.swap_and_mint(theta * x_0)
        print(l)
        x_T = POOLS.total_coins_of_x(l, batch_size, kappa, p, sigma, T)
        log_ret  = np.log(x_T) - np.log(x_0)
        qtl   = np.quantile(-log_ret, alpha)
        return np.mean(-log_ret[-log_ret >= qtl])


pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)
print(pools.SGD())