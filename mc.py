import numpy as np
import copy

def pos_or_zero(x):
    return np.vectorize(max)(x,0)

def fullner(kappa,theta,xi):
    return 2*kappa*theta > xi**2

class Generator:
    def __init__(self, params):
        self.params = params
    def generate(self, num_paths, num_steps, dt):
        return np.zeros((num_paths,num_steps))
    def pertubate(self, param, to, shift='ovr'):
        new_gen = copy.deepcopy(self)
        if shift == 'ovr':
            new_gen.params[param] = to
        elif shift == 'abs':
            new_gen.params[param] += to
        elif shift == 'rel':
            new_gen.params[param] *= (1+to)
        return new_gen

class GBM_gen(Generator):
    # Required params: S0,r,q,sigma
    def __init__(self, params):
        Generator.__init__(self, params)
    def generate(self, num_paths, num_steps, dt, rands=[]):
        paths = Generator.generate(self, num_paths, num_steps, dt)
        S0,r,q,sigma = self.params['S0'],self.params['r'],self.params['q'],self.params['sigma']

        noise = sigma*np.sqrt(dt)*np.random.normal(size=(num_paths,num_steps)) if len(rands) == 0 else rands
        paths[:,0] = S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + noise[:,t])
        return paths

class GBMJD_gen(Generator):
    # Required params: S0,r,q,sigma,j_mu,lambda
    def __init__(self, params):
        Generator.__init__(self, params)
    def generate(self, num_paths, num_steps, dt):
        paths = Generator.generate(self, num_paths, num_steps, dt)

        S0,r,q,sigma = self.params['S0'],self.params['r'],self.params['q'],self.params['sigma']
        jump,lam = self.params['j_mu'],self.params['lambda']

        noise = sigma*np.sqrt(dt)*np.random.normal(size=(num_paths,num_steps))
        poiss = np.random.poisson(lam, size=(num_paths,num_steps))
        paths[:,0] = S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + noise[:,t] + jump*poiss[:,t])
        return paths

class OUJD_gen(Generator):
    # Required params: S0,mu,theta,sigma,j_mu,lambda
    def __init__(self, params):
        Generator.__init__(self, params)
    def generate(self, num_paths, num_steps, dt):
        paths = Generator.generate(self, num_paths, num_steps, dt)

        S0,mu,theta,sigma = self.params['S0'],self.params['mu'],self.params['theta'],self.params['sigma']
        jump,lam = self.params['j_mu'],self.params['lambda']

        noise = sigma*np.sqrt(dt)*np.random.normal(size=(num_paths,num_steps))
        poiss = np.random.poisson(lam, size=(num_paths,num_steps))
        paths[:,0] = S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1] + theta*(mu - paths[:,t-1])*dt + noise[:,t] + jump*poiss[:,t]
        return paths

class Heston_gen(Generator):
    # Required params: S0,r,q,nu0,kappa,theta,xi,rho
    def __init__(self, params):
        Generator.__init__(self, params)
    def generate(self, num_paths, num_steps, dt, rands=[]):
        paths = Generator.generate(self, num_paths, num_steps, dt)

        S0,r,q,nu0 = self.params['S0'],self.params['r'],self.params['q'],self.params['nu0']
        kappa,theta,xi,rho = self.params['kappa'],self.params['theta'],self.params['xi'],self.params['rho']

        dW_S  = np.random.normal(size=(num_paths,num_steps)) if len(rands) == 0 else rands[0]
        dW_X  = np.random.normal(size=(num_paths,num_steps)) if len(rands) == 0 else rands[1]
        dW_nu = rho*dW_S + np.sqrt(1 - rho**2)*dW_X

        nu = np.zeros((num_paths,num_steps))
        nu[:,0] = nu0
        paths[:,0] = S0
        for t in range(1,num_steps):
            nu[:,t] = nu[:,t-1] + kappa*(theta - pos_or_zero(nu[:,t-1]))*dt + xi*np.sqrt(pos_or_zero(nu[:,t-1])*dt)*dW_nu[:,t]
            paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + np.sqrt(pos_or_zero(nu[:,t])*dt)*dW_S[:,t])
        return paths

class Option:
    def __init__(self, strike, expiry, PC, AE, val_params):
        self.strike = strike
        self.expiry = expiry
        self.PC = PC
        self.AE = AE
        self.val_params = val_params

    def payoff(self, S_T):
        return np.maximum(self.strike - S_T, 0) if self.PC == 'P' else np.maximum(S_T - self.strike, 0)

    def value(self, gen=None, paths=None):
        if gen is None and paths is None:
            raise Exception('either MCgen or paths required')

        if paths is None:
            num_paths = self.val_params['num_paths']
            num_steps = self.val_params['num_steps']
            dt = self.expiry / num_steps

            paths = gen.generate(num_paths, num_steps, dt)
        values = self.payoff(paths[:,-1])
        return np.exp(-self.val_params['r']*self.expiry)*values.mean() # values.std()/np.sqrt(num_paths)

    def delta(self, gen, bump=0.01):
        value_S = self.value(gen)
        value_Splus = self.value(gen.pertubate('S0',bump,shift='rel'))
        value_Sminus = self.value(gen.pertubate('S0',-bump,shift='rel'))
        return ((value_Splus - value_S)/0.01 + (value_S - value_Sminus)/0.01)*0.5

    def time_series(self, gen, path):
        values = np.zeros(len(path))
        deltas = np.zeros(len(path))
        for t in range(len(path)):
            new_gen = gen.pertubate('S0',path[t],shift='ovr')
            values[t] = self.value(new_gen)
            deltas[t] = self.value(new_gen)
            self.val_params['num_steps'] -= 1
        return values, deltas
