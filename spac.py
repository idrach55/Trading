import time
import requests
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats 
import pickle

import sys
sys.path.append('/Users/isaacdrachman/GitHub/')
from portfolio import risk, analytics
risk.DATA_DIR = '/Users/isaacdrachman/GitHub/Core/data'

from bs4 import BeautifulSoup
from typing import Dict, List, Tuple


def read_keys(service: str):
    api_key, secret_key = open(f'keys/{service}.keys', 'r').read().split('\n')[:2]
    return api_key, secret_key

def get_alpaca_price(symbol: str) -> float:
    api_key, secret_key = read_keys('alpaca')
    headers = {'APCA-API-KEY-ID': api_key, 'APCA-API-SECRET-KEY': secret_key}
    url = f'https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest'
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise Exception(f'r.status_code = {r.status_code}, r.text = {r.text}')
    return r.json()['trade']['p']

def get_alpaca_prices(symbols: List[str]) -> List[float]:
    prices = []
    for symbol in symbols:
        if symbol[-2:] == 'WT':
            symbol = symbol[:-3] + '.WS'
        try:
            prices.append(get_alpaca_price(symbol))
        except:
            print('symbol not found: {}'.format(symbol))
            prices.append(np.nan)
    return prices

 
def xgb_price(spot: float, vol: float, mdl=None):
    # Model trained on term=5.0, r=0.0080, q=0.00
    mdl = mdl if mdl is not None else pickle.load(open('warrant-mdl-noisy.obj','rb'))
    bs = bs_price(spot, 5.0, vol)
    return mdl.predict(pd.DataFrame({'spot': spot, 'vol': vol},index=[0]))[0] * bs


def xgb_delta(spot: float, vol: float, mdl=None, bump=0.01):
    # Delta shares.
    mdl = mdl if mdl is not None else pickle.load(open('warrant-mdl-noisy.obj','rb'))
    px_up = mdl.predict(pd.DataFrame({'spot': spot*(1.0 + bump), 'vol': vol},index=[0]))[0] * bs_price(spot*(1.0 + bump), 5.0, vol)
    px_dn = mdl.predict(pd.DataFrame({'spot': spot*(1.0 - bump), 'vol': vol},index=[0]))[0] * bs_price(spot*(1.0 - bump), 5.0, vol)
    return (px_up - px_dn) / (2 * bump) / spot


def xgb_vega(spot: float, vol: float, mdl=None):
    mdl = mdl if mdl is not None else pickle.load(open('warrant-mdl-noisy.obj','rb'))
    px_up = mdl.predict(pd.DataFrame({'spot': spot, 'vol': vol + 0.01},index=[0]))[0] * bs_price(spot, 5.0, vol + 0.01)
    px_dn = mdl.predict(pd.DataFrame({'spot': spot, 'vol': vol - 0.01},index=[0]))[0] * bs_price(spot, 5.0, vol - 0.01)
    return (px_up - px_dn) / 2


def xgb_iv(price: float, spot: float):
    # Model trained on term=5.0, r=0.0080, q=0.00
    mdl = pickle.load(open('warrant-mdl-noisy.obj','rb'))
    def internal(vol):
        return xgb_price(spot, vol, mdl=mdl) - price
    # Snap to bounds if vol falls outside.
    if internal(0.08) > 0:
        return 0.08
    if internal(1.00) < 0:
        return 1.00
    return opt.brentq(internal, 0.08, 1.00)


def mc_paths(spot: float, term: float, vol: float, noise: np.array, r: float = 0.0080, q: float = 0.00) -> np.array:
    return spot * (1.0 + (r - q)/252.0 + vol*np.sqrt(1.0/252.0)*noise).cumprod(axis=1)


def mc_price(spot: float, term: float, vol: float, noise: np.array, r: float = 0.0080, q: float = 0.00, 
             strike: float = 11.5, ko: float = 18.0) -> float:
    """
    Price warrant using monte-carlo as a daily up-and-out call.
    """
    
    paths = mc_paths(spot, term, vol, noise, r=r, q=q)

    pvs = np.zeros(paths.shape[0])
    for idx_path, idx_ko in enumerate(np.argmax(paths >= ko, axis=1)):
        # If path starts at/above knock-out.
        if idx_ko == 0 and paths[idx_path][idx_ko] >= ko:
            pvs[idx_path] = paths[idx_path][idx_ko] - strike
        # If path never knocks-out.
        elif idx_ko == 0:
            pvs[idx_path] = np.exp(-r * term) * max(paths[idx_path][-1] - strike, 0.0)
        # If path knocks-out.
        else:
            pvs[idx_path] = np.exp(-r * idx_ko/252) * paths[idx_path][idx_ko] - strike
    return pvs


def mc_price2(spot: float, term: float, vol: float, noise: np.array, r: float = 0.0080, q: float = 0.00,
              strike: float = 11.5, ko: float = 18.0, ko_days: int = 20, ko_window: int = 30):
    paths = mc_paths(spot, term, vol, noise, r=r, q=q)
    """
    Price warrant using monte-carlo with more accurate knock-out condition, i.e. 20 days out of 30 above barrier.

    WARN: This is super slow. Need to do better, even for training xgb.
    """

    pvs = np.zeros(paths.shape[0])
    for idx_path in range(paths.shape[0]):
        # If this has no chance of knocking-out (ie < 20 days above KO), then cut to expiry valuation.
        if (paths[idx_path] >= ko).sum() < ko_days:
            pvs[idx_path] = np.exp(-r * term) * max(paths[idx_path][-1] - strike, 0.0)
            continue
    
        # Otherwise scan 30 day windows.
        for idx_step in range(ko_window-1, paths.shape[1]):
            if (paths[idx_path][idx_step - ko_window + 1:idx_step+1] >= ko).sum() >= ko_days:
                pvs[idx_path] = np.exp(-r * idx_step/252.0) * max(paths[idx_path][idx_step] - strike, 0.0)
                break
            if idx_step == paths.shape[1] - 1:
                pvs[idx_path] = np.exp(-r * term) * max(paths[idx_path][idx_step] - strike, 0.0)
    return pvs
    

def bs_price(spot: float, term: float, vol: float, r: float = 0.0080, q: float = 0.00, strike: float = 11.5) -> float:
    d1 = (np.log(spot / strike) + (r + vol**2/2)*term)/(vol*np.sqrt(term))
    d2 = d1 - vol*np.sqrt(term)
    return stats.norm.cdf(d1) * spot - stats.norm.cdf(d2) * strike * np.exp(-r * term)


def mc_iv(price: float, spot: float, term: float, noise: np.array, r: float = 0.0080, q: float = 0.00, 
          strike: float = 11.5, ko: float = 18.0) -> float:
    def internal(vol):
        return mc_price(spot, term, vol, noise, r=r, q=q, strike=strike, ko=ko).mean() - price
    return opt.brentq(internal, 0.01, 1.50)


def read_positions(fname: str):
    positions = pd.read_csv(fname)
    positions = positions.loc[positions.Description.apply(lambda desc: 'WTS' in desc if type(desc) == str else False)]
    positions = positions.assign(Warrant=list(map(lambda ticker: ticker[:-2]+'-WT' if ticker[-2:] == 'WS' else ticker, positions.Ticker)))
    positions = positions.assign(Common=list(map(lambda ticker: ticker[:-2] if ticker[-2:] == 'WS' else ticker[:-1], positions.Ticker)))
    positions = positions.assign(Shares=list(map(lambda desc: 0.5 if 'half' in desc.lower() or '1/2' in desc else 1.0, positions.Description)))
    df = positions[['Common','Warrant','Quantity','Cost','Shares']]
    
    # Override common symbol for ACKI and shares for ESSC. VHAQ should be handled as "HALF" is in description.
    df.loc[df.loc[df.Common == 'ACKI'].index, 'Common'] = 'ACKIT'
    df.loc[df.loc[df.Common == 'ESSC'].index, 'Shares'] = 0.5
    #df.loc[df.loc[df.Common == 'VHAQ'].index, 'Shares'] = 0.5

    df.to_csv('warrant-folio.csv', index=False)


def analyze_warrant(symbol: str, mc=False):
    common_px = get_alpaca_price(symbol)
    try:
        warrant_px = get_alpaca_price(symbol+'W')
    except:
        warrant_px = get_alpaca_price(symbol+'.WS')
    if mc:
        noise = np.random.normal(size=(20000, 5*252))
        vol = 100.0 * mc_iv(warrant_px, common_px, 5, noise) 
        target = mc_price(common_px, 5, 0.35, noise).mean()
    else:
        vol = 100.0 * xgb_iv(warrant_px, common_px)
        target = xgb_price(common_px, 0.35)
    profit_pct = 100.0 * (target / warrant_px - 1.0)
    deal_prob = 100.0 * warrant_px / target
    print('Px: {:.2f} IV: {:0.1f}% Target Px: {:.2f} Profit: {:.0f}% Deal Prob. {:.0f}%'.format(warrant_px, vol, target, profit_pct, deal_prob))


def warrant_folio(fname: str = 'warrant-folio.csv', mc=False):
    folio = pd.read_csv(fname, index_col=0)
    
    # Unless specified as 1/2 share per warrant, set to 1.  
    folio.Shares = folio.Shares.fillna(1.0)
    folio = folio.assign(Common_Px = get_alpaca_prices(folio.index), Warrant_Px = get_alpaca_prices(folio.Warrant))
    folio = folio.assign(Value = folio.Warrant_Px * folio.Quantity)
    folio = folio.assign(**{'PL': folio.Value - folio.Cost, 'PL%': 100.0 * folio.Value/folio.Cost - 100.0})
    
    vols = []
    deltas = []
    vegas = []
    targets = []
    
    for _, row in folio.iterrows():
        # Re-generate random noise for each name so correlation is not 1.
        if mc:
            noise = np.random.normal(size=(20000, 5*252))
        # Default $18 KO, 5 years to expiry, 80bps risk-free rate, and 0bps borrow cost.
        if mc:
            vols.append(100.0 * mc_iv(row.Warrant_Px / row.Shares, row.Common_Px, 5, noise))
            targets.append(mc_price(row.Common_Px, 5, 0.35, noise).mean())
        else:
            vols.append(100.0 * xgb_iv(row.Warrant_Px / row.Shares, row.Common_Px))
            targets.append(xgb_price(row.Common_Px, 0.35))
            deltas.append(row.Quantity * row.Shares * row.Common_Px * xgb_delta(row.Common_Px, vols[-1] / 100.0))
            vegas.append(row.Quantity * row.Shares * xgb_vega(row.Common_Px, vols[-1] / 100.0))

    folio = folio.assign(IV = vols, Target = targets, Delta = deltas, Vega = vegas)
    # Target P&L is from current value, not cost.
    folio = folio.assign(**{'Target PL': folio.Quantity * folio.Target - folio.Value, 
                            'Target PL%': 100.0 * folio.Target / (folio.Warrant_Px / folio.Shares) - 100.0,
                            'Implied Deal': 100.0 * (folio.Warrant_Px / folio.Shares) / folio.Target})
    return folio


def clean_df() -> pd.DataFrame:
    """
    Read the SPAC spreadsheet and clean up the names of some of the columns for easier use.
    Pre-process tags into list instead of string.
    """

    df = pd.read_csv('SPAC.csv', index_col=0)
    df.index = df['Shares Symbol']
    df.columns = list(map(lambda col: col.replace('\xa0 ', ' '), df.columns))
    replacements = {'Initial Size (in millions)': 'Initial Size',
                    'Approx. Age (in months)': 'Age',
                    'Remaining (in months)': 'Remaining'}
    for k, v in replacements.items():
        df[v] = df[k]
        df = df.drop(k, axis=1)
    df.Tags = list(map(lambda tags: tags.split(', ') if type(tags) == str else tags, df.Tags))
    return df


def print_stats(df: pd.DataFrame):
    for stage in np.sort(df.Stage.unique()):
        stage_str = '{:<20} min: {:>5.2f} max: {:>5.2f} mean: {:>5.2f} stddev: {:>5.2f} count: {:>3d}'
        stage_prices = df.loc[df.Stage == stage]['Shares Last Close']
        print(stage_str.format(stage,stage_prices.min(),stage_prices.max(),stage_prices.mean(),stage_prices.std(),len(stage_prices)))


def has_results(symbol: str) -> bool:
    """
    Check Yahoo Finance for a symbol.
    """
    url = 'https://finance.yahoo.com/quote/{}?p={}'.format(symbol, symbol)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features='lxml')
    if soup.find('title').get_text() == 'Symbol Lookup from Yahoo Finance':
        return False
    return True


def get_yf_price(symbol: str) -> float:
    url = 'https://finance.yahoo.com/quote/{}?p={}'.format(symbol, symbol)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features='lxml')
    spans = soup.find_all('span', class_='Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)')
    if len(spans) == 0 or spans[0].get_text() == '':
        return None
    return float(spans[0].get_text())


def lookup_warrant(symbol: str) -> str:
    """
    Confirm whether warrant is titled {}W or {}-WT for this symbol.
    """

    # First, check data directory so we don't need YF.
    if risk.is_symbol_cached(symbol+'W')[0] != -1:
        return symbol+'W'
    elif risk.is_symbol_cached(symbol+'-WS')[0] != -1:
        return symbol+'-WS'

    # If neither found, check Yahoo Finance.
    if has_results(symbol+'W'):
        return symbol+'W'
    elif has_results(symbol+'-WT'):
        # Note: Yahoo uses -WT but AlphaVantage uses -WS.
        return symbol+'-WS'
    return None


def get_warrants(symbols: List[str]) -> pd.DataFrame:
    """
    Determine warrant symbol for each and download prices.
    """

    warrants = []
    for symbol in symbols:
        warrant = lookup_warrant(symbol)
        if warrant is None:
            continue
        warrants.append(warrant)
    prices = get_prices(warrants)
    prices.columns = list(map(lambda ticker: ticker[:-1] if ticker[-1] == 'W' else ticker[:-3], prices.columns))
    return prices


def get_prices(symbols: List[str], existing_prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Get prices for list of symbols, using 1m delays after 5 requests in a row,
    and ignore tickers already found in another dataframe (if provided.)
    """

    prices = {}
    count = 0
    for symbol in symbols:
        if existing_prices is not None and symbol in existing_prices.columns:
            continue
        if count == 5:
            time.sleep(60)
            count = 0
        if risk.is_symbol_cached(symbol)[0] == -1:
            count += 1
        try:
            prices[symbol] = risk.get_prices(risk.get_data([symbol], data_age_limit=30))[symbol]
        except:
            continue
    return pd.concat(prices,axis=1)


def get_prices_from_data(data: pd.DataFrame, existing_prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Download prices based on tickers in dataframe from spactrax.
    Most are listed under 'Shares Symbol' field, but some post-merger names will have earlier history
    under the prior 'Other Symbol'.
    """

    if existing_prices is None:
        return get_prices(list(data['Shares Symbol']))
    else:
        return get_prices(list(data['Other Symbol']), existing_prices=existing_prices)


def align_data(data: pd.DataFrame, prices: pd.DataFrame, t_pre: int, t_pos: int, tag: str) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    For names that have merged, find price history pre and post merger for certain number of days.
    Align all the names into same dataframe, ie price as of day t where merger closed day t_pre+1.
    """

    dates = data[tag]
    dates.index = data['Shares Symbol']
    dates = dates.dropna()

    val_after_event = {}
    all_names = pd.DataFrame()
    for symbol, date in dates.items():
        other_symbol = data.loc[data['Shares Symbol'] == symbol]['Other Symbol'].iloc[0]
        if other_symbol not in prices.columns:
            other_symbol = symbol
        pre = prices[symbol][:dates[symbol]].dropna().iloc[-t_pre-1:]
        pre_other = prices[other_symbol][:dates[symbol]].dropna().iloc[-t_pre-1:]
        pre = pre_other if len(pre_other) > len(pre) else pre
        pos = prices[symbol][dates[symbol]:].dropna().iloc[:t_pos]
        if len(pre)+len(pos) < t_pre+t_pos+1:
            continue
        series = pre.append(pos)
        series.name = symbol
        val_after_event[symbol] = series
        all_names[symbol] = val_after_event[symbol].reset_index()[symbol]
    return val_after_event, all_names
