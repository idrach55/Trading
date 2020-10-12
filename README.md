# Trading

Strategy Backtests
- Jupyter notebooks for simple payoff backtests (i.e. no m2m)
- capped/uncapped and outright variance selling <code>strat_variance_seller.ipynb</code>
- 1m variance selling using daily vix for entry level and m2m <code>varswap_with_m2m.ipynb</code>  
- palladium (call on dispersion) <code>strat_palladium.ipynb</code>

Other Research
- Predicting realized variance spreads using linear reg or simple neural net <code>relative_value.ipynb</code>
- Looking for dislocations of SX5E dividend futures and dividend term structure as a result of structured products <code>strat_sx5edivs.ipynb</code>
- Modeling empirical non-normality of one-day stock returns <code>stocks.ipynb</code>

Monte Carlo
- vanilla option pricing using monte carlo where underlying process can be non-GBM, <code>mc.py</code> and <code>mc_example.ipynb</code>
- processes include GBM (Brownian motion), GBMJD (Brownian motion + jump), OU (mean-reverting), OUJD (mean-reverting + jump)
