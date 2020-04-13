
"""
Module for building a complete daily dataset from Quandl's WIKI dataset.
"""
from io import BytesIO
import tarfile
from zipfile import ZipFile

from click import progressbar
from logbook import Logger
import pandas as pd
import requests
from six.moves.urllib.parse import urlencode
from six import iteritems
from trading_calendars import register_calendar_alias

from zipline.utils.deprecate import deprecated
from zipline import core as bundles
from zipline.data.hdf5_daily_bars import HDF5DailyBarWriter, HDF5DailyBarReader
import numpy as np
"""
Module for building a complete daily dataset from Quandl's WIKI dataset.
"""
from io import BytesIO
import tarfile
from zipfile import ZipFile

from click import progressbar
from logbook import Logger
import pandas as pd
import requests
from six.moves.urllib.parse import urlencode
from six import iteritems
from trading_calendars import register_calendar_alias

from zipline.utils.deprecate import deprecated

import numpy as np

log = Logger(__name__)



@bundles.register('hdfbundle')
def quandl_bundle(environ,
                  asset_db_writer,
                  minute_bar_writer,
                  daily_bar_writer,
                  adjustment_writer,
                  calendar,
                  start_session,
                  end_session,
                  cache,
                  show_progress,
                  output_dir):
    """
    quandl_bundle builds a daily dataset using Quandl's WIKI Prices dataset.

    For more information on Quandl's API and how to obtain an API key,
    please visit https://docs.quandl.com/docs#section-authentication
    """
    api_key = environ.get('QUANDL_API_KEY')
    if api_key is None:
        raise ValueError(
            "Please set your QUANDL_API_KEY environment variable and retry."
        )

    raw_data = fetch_data()


    asset_metadata = gen_asset_metadata(
        raw_data[['symbol', 'date']],
        show_progress
    )
    asset_db_writer.write(asset_metadata)

    symbol_map = asset_metadata.symbol
    sessions = calendar.sessions_in_range(start_session, end_session)

    raw_data.set_index(['date', 'symbol'], inplace=True)
    daily_bar_writer.write(
        parse_pricing_and_vol(
            raw_data,
            sessions,
            symbol_map
        ),
        show_progress=show_progress
    )

    raw_data.reset_index(inplace=True)
    raw_data['symbol'] = raw_data['symbol'].astype('category')
    raw_data['sid'] = raw_data.symbol.cat.codes
    adjustment_writer.write(
        splits=parse_splits(
            raw_data[[
                'sid',
                'date',
                'split_ratio',
            ]].loc[raw_data.split_ratio != 1],
            show_progress=show_progress
        ),
        dividends=parse_dividends(
            raw_data[[
                'sid',
                'date',
                'ex_dividend',
            ]].loc[raw_data.ex_dividend != 0],
            show_progress=show_progress
        )
    )





def fetch_data(*args, **kwargs  ):

    """
        Fetch data from database
    """

    try:
        return pd.DataFrame(columns=[  'symbol',
                    'date',
                    'open',
                    'high',
                    'low',
                    'close',
                    'volume',
                    'ex_dividend',
                    'split_ratio',])

    except Exception:
        log.exception("Exception in fecthing data")




def gen_asset_metadata(data):


    data = data.groupby( by='symbol' ).agg({'date': ['min', 'max']}).\
          rename(columns={'min':'start_date','max':'end_date'}).\
          assign(
        exchange='QUANDL',
        auto_close_date=lambda x:x['end_date'].values + pd.Timedelta(days=1)).\
         reset_index()
    data.columns = data.columns.get_level_values(0)
    return data


def parse_splits(data, show_progress):
    if show_progress:
        log.info('Parsing split data.')

    data['split_ratio'] = 1.0 / data.split_ratio
    data.rename(
        columns={
            'split_ratio': 'ratio',
            'date': 'effective_date',
        },
        inplace=True,
        copy=False,
    )
    return data


def parse_dividends(data, show_progress):
    if show_progress:
        log.info('Parsing dividend data.')

    data['record_date'] = data['declared_date'] = data['pay_date'] = pd.NaT
    data.rename(
        columns={
            'ex_dividend': 'amount',
            'date': 'ex_date',
        },
        inplace=True,
        copy=False,
    )
    return data


def parse_pricing_and_vol(data,
                          sessions,
                          symbol_map):
    for asset_id, symbol in iteritems(symbol_map):
        asset_data = data.xs(
            symbol,
            level=1
        ).reindex(
            sessions.tz_localize(None)
        ).fillna(0.0)
        yield asset_id, asset_data


register_calendar_alias("CSVDIR", "NYSE")
