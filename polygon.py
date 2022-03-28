"""
Author: Isaac Drachman
Date:   3/5/2022
"""

import requests
import pandas as pd

from typing import List, Dict

class REST:
    def __init__(self, keys: str = 'keys/polygon.keys'):
        self.root_url = 'https://api.polygon.io'
        self.api_key = open(keys, 'r').read()

    def _get_response(self, endpoint: str, params: Dict[str,str] = {}):
        # Concatenate URL together and add api_key at the end.
        url = f'{self.root_url}{endpoint}?'
        for key, value in params.items():
            url += f'{key}={value}&'
        url += f'apiKey={self.api_key}'

        # Request data and error catching.
        r = requests.get(url)
        content = r.json()
        if content['status'] == 'ERROR':
            raise Exception(content['error'])

        results = content['results']
        # Paginate and merge.
        while 'next_url' in content:
            r = requests.get(content['next_url']+f'&apiKey={self.api_key}')
            content = r.json()
            results += content['results']
        return results

    def tickers(self, exchanges: List[str] = None) -> List[dict]:
        endpoint = '/v3/reference/tickers'
        params = {'type': 'CS',
                  'market': 'stocks',
                  'active': 'true',
                  'sort': 'ticker',
                  'limit': 1000}
        if exchanges is None:
            return self._get_response(endpoint, params)
        else:
            results = []
            for exchange in exchanges:
                params['exchange'] = exchange
                results += self._get_response(endpoint, params)
            return results

    def ticker_details(self, ticker: str) -> dict:
        endpoint = f'/v3/reference/tickers/{ticker}'
        return self._get_response(endpoint)

    def dividends(self, ticker: str, dividend_type: str = '') -> List[dict]:
        endpoint = '/v3/reference/dividends'
        params = {'ticker': ticker,
                  'dividend_type': dividend_type,
                  'limit': 1000,
                  'sort': 'ex_dividend_date',
                  'order': 'asc'}
        df = pd.DataFrame(self._get_response(endpoint, params))
        if len(df) > 0: 
            df.ex_dividend_date = pd.to_datetime(df.ex_dividend_date)
        return df

    def aggs(self, ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str, adjusted = True, limit: int = 5000):
        """
        timespan: minute, hour, day, week, month, quarter, year
        """
        endpoint = f'/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'
        params = {'adjusted': adjusted, 'limit': limit}
        results = self._get_response(endpoint, params)
        df = pd.DataFrame(results)
        df.index = pd.to_datetime(df.t * 1e6)
        return df.drop('t', axis=1)