"""User-requested nominal-price policy: no minimum or maximum share price."""
import numpy as np
import pandas as pd

POLICY = 'VALID_POSITIVE_PRICE_NO_NOMINAL_BAND_20261017'

def valid_price(values):
    """Keep data validity checks without excluding cheap or expensive shares."""
    price = pd.to_numeric(values, errors='coerce')
    return np.isfinite(price) & price.gt(0)
