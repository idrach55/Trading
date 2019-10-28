import numpy as np
import copy

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
    def __init__(self, params):
        Generator.__init__(self, params)
    def generate(self, num_paths, num_steps, dt):
        paths = Generator.generate(self, num_paths, num_steps, dt)
        S0,r,q,sigma = self.params['S0'],self.params['r'],self.params['q'],self.params['sigma']
        noise = sigma*np.sqrt(dt)*np.random.normal(size=(num_paths,num_steps))
        paths[:,0] = S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + noise[:,t])
        return paths

class OU_gen(Generator):
    def __init__(self, params):
        Generator.__init__(self, params)
    def generate(self, num_paths, num_steps, dt):
        pass

class GBMJD_gen(Generator):
    def __init__(self, params):
        Generator.__init__(self, params)
    def generate(self, num_paths, num_steps, dt):
        paths = Generator.generate(self, num_paths, num_steps, dt)

        S0,r,q,sigma = self.params['S0'],self.params['r'],self.params['q'],self.params['sigma']
        j_mu,j_sigma,lam = self.params['j_mu'],self.params['j_sigma'],self.params['lambda']

        noise = sigma*np.sqrt(dt)*np.random.normal(size=(num_paths,num_steps))
        jump  = j_mu + j_sigma*np.random.normal(size=(num_paths,num_steps))
        poiss = np.random.poisson(lam, size=(num_paths,num_steps))
        paths[:,0] = S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + noise[:,t] + jump[:,t]*poiss[:,t])
        return paths

class OUJD_gen(Generator):
    def __init__(self, params):
        Generator.__init__(self, params)
    def generate(self, num_paths, num_steps, dt):
        paths = Generator.generate(self, num_paths, num_steps, dt)

        S0,mu,theta,sigma = self.params['S0'],self.params['mu'],self.params['theta'],self.params['sigma']
        j_mu,j_sigma,lam = self.params['j_mu'],self.params['j_sigma'],self.params['lambda']

        noise = sigma*np.sqrt(dt)*np.random.normal(size=(num_paths,num_steps))
        jump  = j_mu + j_sigma*np.random.normal(size=(num_paths,num_steps))
        poiss = np.random.poisson(lam, size=(num_paths,num_steps))
        paths[:,0] = S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1] + theta*(mu - paths[:,t-1])*dt + noise[:,t] + jump[:,t]*poiss[:,t]
        return paths

def MC(gen, num_paths, num_steps, dt, payoff):
    if type(gen) == type(np.sum):
        paths = gen(num_paths, num_steps, dt)
    else:
        paths = gen.generate(num_paths, num_steps, dt)
    f = np.vectorize(payoff)
    values = f(paths[:,-1])
    return values.mean(), values.std()/np.sqrt(num_paths)

class Option:
    def __init__(self, strike, expiry, PC, AE, val_params):
        self.strike = strike
        self.expiry = expiry
        self.PC = PC
        self.AE = AE
        self.val_params = val_params

    def payoff(self, S_T):
        return max(self.strike - S_T, 0) if self.PC == 'PUT' else max(S_T - self.strike, 0)

    def time_series(self, gen, path):
        values = np.zeros(len(path))
        deltas = np.zeros(len(path))
        for t in range(len(path)):
            new_gen = gen.pertubate('S0',path[t],shift='ovr')
            values[t] = self.value(new_gen)
            deltas[t] = self.value(new_gen)
            self.val_params['num_steps'] -= 1
        return values, deltas

    def value(self, gen):
        num_paths = self.val_params['num_paths']
        num_steps = self.val_params['num_steps']
        dt = self.val_params['dt']
        mean_value, error = MC(gen, num_paths, num_steps, dt, self.payoff)
        return mean_value

    def delta(self, gen, bump=0.01):
        value_S = self.value(gen)
        value_Splus = self.value(gen.pertubate('S0',bump,shift='rel'))
        value_Sminus = self.value(gen.pertubate('S0',-bump,shift='rel'))
        return ((value_Splus - value_S)/0.01 + (value_S - value_Sminus)/0.01)*0.5
