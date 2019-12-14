import pandas as pd
import numpy as np
import scipy.optimize as opt
import mc

# Read csv data
def read_data(filename):
    options = pd.read_csv(filename)

    # Accounting rate in %, and compute the cost-of-carry (i.e. div+borrow)
    # Compute mid-price for each option
    options.Rate /= 100
    options['Q'] = options.Rate - np.log(options.Fwd/options.Spot)/options.TTE
    options['Mid'] = 0.5*(options['Bid'] + options['Ask'])

    # Remove ITM options and those with no-bid
    options = options.loc[(options.Bid > 0.0) & (((options.Strike > options.Spot) & (options.CallPut == 'C')) |
                          ((options.Strike < options.Spot) & (options.CallPut == 'P')))]
    return options

# Build mc.Option objects out of each option
def data_to_options(data):
    return [mc.Option(row.Strike, row.TTE, row.CallPut, 'E', {'r':row.Rate}) for idx,row in data.iterrows()]

# Slice for given date/strikes
def select_options(options, date, strikes):
    return options.loc[(options.Date == date) & (options.Strike.isin(strikes))]

class Calibrator:
    def __init__(self, options, val_params, gen_class):
        self.options = options
        self.val_params = val_params
        self.gen_class = gen_class

        # Generate all the random variables upfront to speed up computation
        # So for each iteration of new model params, same initial random variables are used
        # However, this is model-dependent
        if self.gen_class == mc.Heston_gen:
            self.rands = np.random.normal(size=(2,self.val_params['num_paths'],self.val_params['num_steps']))
        else:
            self.rands = []

    # updates: list/array of new params
    # update_keys: list of keys for each param being updated in 'updates'
    # model: dictionary of model parameters (static and those being updated)
    def evaluate(self, updates, update_keys, model, fixrands=True):
        num_paths = self.val_params['num_paths']
        num_steps = self.val_params['num_steps']

        # Assumes all options are of the same expiry
        dt = self.options[0].expiry / num_steps

        # Update model params we are calibrating
        for idx,key in enumerate(update_keys):
            model[key] = updates[idx]

        # Value all options on the same set of paths for performance
        gen = self.gen_class(model)
        if fixrands:
            paths = gen.generate(num_paths,num_steps,dt,rands=self.rands)
        else:
            paths = gen.generate(num_paths,num_steps,dt)
        values = np.array([mc_opt.value(paths=paths) for mc_opt in self.options])
        return values

    def loss(self, updates, update_keys, model, market, fixrands=True):
        px = self.evaluate(updates, update_keys, model, fixrands=fixrands)
        # Use the average absolute percentage error
        return np.mean(np.abs(100*(px/market - 1)))

    def heston_bounds(self):
        return [(0.01**2,0.50**2),(0,10),(0.01**2,0.50**2),(0.10,1.50),(-1,0)]

    # Define the Fullner constraint to ensure positive variance
    def heston_cons(self):
        return [{'type':'ineq', 'fun': lambda x: 2*x[1]*x[2] - x[3]**2}]

    #
    def run_GD(self,x0,model,update_keys,market,maxiter=50):
        bounds = self.heston_bounds()
        cons   = self.heston_cons()

        def f(updates):
            return self.loss(updates,update_keys,model,market)
        res = opt.minimize(f, x0, bounds=bounds, constrains=cons, options={'maxiter':maxiter})
        return res['x']

    # bounds: list of 2-tuples for min/max of each parameter being fit
    def run_DE(self,model,update_keys,market,maxiter=50,popsize=15):
        bounds = self.heston_bounds()

        def f(updates):
            return self.loss(updates,update_keys,model,market)

        # run scipy differential_evolution, using multiprocessing
        res = opt.differential_evolution(f, bounds, maxiter=maxiter, popsize=popsize, disp=True, updating='deferred', workers=-1)
        return res['x']

def example():
    # Read data and select specific day & strikes to calibrate off of
    data = read_data('option_px.csv')
    selection = select_option(data, '2019-12-06', [7000, 7500, 8000, 8250, 8500, 8750, 9000])
    options = data_to_options(selection)

    # Pull spot, risk-free, cost-of-carry
    # We don't need to specify initial Heston params when using DE
    model = {'S0': selection.Spot.iloc[0], 'r': selection.Spot.Rate, 'q': selection.Spot.Q}

    # Initialize Heston calibrator
    fit_heston = Calibrator(options, {'num_paths': 25000, 'num_steps': 100}, mc.Heston_gen)
    update_keys = ['nu0','kappa','theta','xi','rho']

    # Initial guess
    x0 = [0.17**2, 1.0, 0.20**2, 0.40, -0.90]

    # Fit
    res = fit_GD(x0,model,update_keys,selection.Mid.values)
