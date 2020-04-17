"""
HDF5 Pricing File Format
------------------------
At the top level, the file is keyed by country (to support regional
files containing multiple countries).

Within each country, there are 4 subgroups:

``/data``
^^^^^^^^^
Each field (OHLCV) is stored in a dataset as a 2D array, with a row per
sid and a column per session. This differs from the more standard
orientation of dates x sids, because it allows each compressed block to
contain contiguous values for the same sid, which allows for better
compression.

.. code-block:: none

   /data
     /open
     /high
     /low
     /close
     /volume

``/index``
^^^^^^^^^^
Contains two datasets, the index of sids (aligned to the rows of the
OHLCV 2D arrays) and index of sessions (aligned to the columns of the
OHLCV 2D arrays) to use for lookups.

.. code-block:: none

   /index
     /sid
     /day

``/lifetimes``
^^^^^^^^^^^^^^
Contains two datasets, start_date and end_date, defining the lifetime
for each asset, aligned to the sids index.

.. code-block:: none

   /lifetimes
     /start_date
     /end_date

``/currency``
^^^^^^^^^^^^^

Contains a single dataset, ``code``, aligned to the sids index, which contains
the listing currency of each sid.

Example
^^^^^^^
Sample layout of the full file with multiple countries.

.. code-block:: none

   |- /US
   |  |- /data
   |  |  |- /open
   |  |  |- /high
   |  |  |- /low
   |  |  |- /close
   |  |  |- /volume
   |  |
   |  |- /index
   |  |  |- /sid
   |  |  |- /day
   |  |
   |  |- /lifetimes
   |  |  |- /start_date
   |  |  |- /end_date
   |  |
   |  |- /currency
   |     |- /code
   |
   |- /CA
      |- /data
      |  |- /open
      |  |- /high
      |  |- /low
      |  |- /close
      |  |- /volume
      |
      |- /index
      |  |- /sid
      |  |- /day
      |
      |- /lifetimes
      |  |- /start_date
      |  |- /end_date
      |
      |- /currency
         |- /code
"""

from collections import namedtuple
from functools import partial

import h5py
import logbook
import numpy as np
import pandas as pd
from six import iteritems, raise_from, viewkeys
from six.moves import reduce

from zipline.data.bar_reader import (
    NoDataAfterDate,
    NoDataBeforeDate,
    NoDataForSid,
    NoDataOnDate,
)
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import bytes_array_to_native_str_object_array
from zipline.utils.pandas_utils import check_indexes_all_same

Field = namedtuple('Field', "name, dtype")
log = logbook.Logger('HDF5DailyBars')

VERSION = 0

DATA = 'data'
INDEX = 'index'
LIFETIMES = 'lifetimes'
CURRENCY = 'currency'
CODE = 'code'

DAY = 'day'
SID = 'sid'

START_DATE = 'start_date'
END_DATE = 'end_date'

# XXX is reserved for "transactions involving no currency".
MISSING_CURRENCY = 'USD'


def convert_price_with_scaling_factor(a, scaling_factor):
    conversion_factor = (1.0 / scaling_factor)

    zeroes = (a == 0)
    return np.where(zeroes, np.nan, a.astype('float64')) * conversion_factor


class HDFReader(CurrencyAwareSessionBarReader):
    """
    Parameters
    ---------
    country_group : h5py.Group
        The group for a single country in an HDF5 daily pricing file.
    """

    def __init__(self, country_group, dataset=None):
        self._country_group = country_group

        self._postprocessors = {
            column.name: partial(convert_price_with_scaling_factor,
                                 scaling_factor=self._read_scaling_factor(column.name))
            for column in dataset

        }

    @classmethod
    def from_file(cls, h5_file, country_code):
        """
        Construct from an h5py.File and a country code.

        Parameters
        ----------
        h5_file : h5py.File
            An HDF5 daily pricing file.
        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        if h5_file.attrs['version'] != VERSION:
            raise ValueError(
                'mismatched version: file is of version %s, expected %s' % (
                    h5_file.attrs['version'],
                    VERSION,
                ),
            )

        return cls(h5_file[country_code])

    @classmethod
    def from_path(cls, path, country_code):
        """
        Construct from a file path and a country code.

        Parameters
        ----------
        path : str
            The path to an HDF5 daily pricing file.
        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        return cls.from_file(h5py.File(path), country_code)

    def _read_scaling_factor(self, field):
        return self._country_group[DATA][field].attrs['scaling_factor']

    def load_raw_arrays(self,
                        columns,
                        start_date,
                        end_date,
                        assets):
        """
        Parameters
        ----------
        columns : list of str
           'open', 'high', 'low', 'close', or 'volume'
        start_date: Timestamp
           Beginning of the window range.
        end_date: Timestamp
           End of the window range.
        assets : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        self._validate_timestamp(start_date)
        self._validate_timestamp(end_date)
        start_ix = self.dates.searchsorted(start_date.asm8)
        stop_ix = self.dates.searchsorted(end_date.asm8, side='right')
        n_dates = stop_ix - start_ix
        n_sids = len(self.sids)
        # Create a buffer into which we'll read data from the h5 file.
        # Allocate an extra row of space that will always contain null values.
        # We'll use that space to provide "data" for entries in ``assets`` that
        # are unknown to us.
        num_rows = n_dates * n_sids
        full_buf = np.zeros((n_sids + 1, n_dates), dtype=np.uint32)

        # We'll only read values into this portion of the read buf.
        mutable_buf = np.zeros(num_rows, dtype=np.uint32)

        # Indexer that converts an array aligned to self.sids (which is what we
        # pull from the h5 file) into an array aligned to ``assets``.
        #
        # Unknown assets will have an index of -1, which means they'll always
        # pull from the last row of the read buffer. We allocated an extra
        # empty row above so that these lookups will cause us to fill our
        # output buffer with "null" values.
        sid_selector = self._make_sid_selector(assets)  # not sure what it is  remove -1 not correct not repeat
        unique_ides = np.unique(np.sort(sid_selector)[1:])

        idx_selection = [
            x + d * n_sids
            for d in range(start_ix, stop_ix)
            for x in unique_ides
        ]
        out = []
        for column in columns:
            # Zero the buffer to prepare to receive new data.
            mutable_buf.fill(0)
            dataset = self._country_group[DATA][column]
            # Fill the mutable portion of our buffer with data from the file.
            dataset.read_direct(
                mutable_buf,
                np.s_[idx_selection],
            )

            mutable_buf.reshape((n_dates, n_sids))
            full_buf[:-1] = mutable_buf

            # Select data from the **full buffer**. Unknown assets will pull
            # from the last row, which is always empty.
            out.append(self._postprocessors[column](full_buf[sid_selector].T))

        return out

    def _make_sid_selector(self, assets):
        """
        Build an indexer mapping ``self.sids`` to ``assets``.

        Parameters
        ----------
        assets : list[int]
            List of assets requested by a caller of ``load_raw_arrays``.

        Returns
        -------
        index : np.array[int64]
            Index array containing the index in ``self.sids`` for each location
            in ``assets``. Entries in ``assets`` for which we don't have a sid
            will contain -1. It is caller's responsibility to handle these
            values correctly.
        """
        assets = np.array(assets)
        sid_selector = self.sids.searchsorted(assets)
        unknown = np.in1d(assets, self.sids, invert=True)
        sid_selector[unknown] = -1
        return sid_selector

    def _validate_assets(self, assets):
        """Validate that asset identifiers are contained in the daily bars.

        Parameters
        ----------
        assets : array-like[int]
           The asset identifiers to validate.

        Raises
        ------
        NoDataForSid
            If one or more of the provided asset identifiers are not
            contained in the daily bars.
        """
        missing_sids = np.setdiff1d(assets, self.sids)

        if len(missing_sids):
            raise NoDataForSid(
                'Assets not contained in daily pricing file: {}'.format(
                    missing_sids
                )
            )

    def _validate_timestamp(self, ts):
        if ts.asm8 not in self.dates:
            raise NoDataOnDate(ts)

    @lazyval
    def dates(self):
        return self._country_group[INDEX][DAY][:].astype('datetime64[ns]')

    @lazyval
    def sids(self):
        return self._country_group[INDEX][SID][:].astype('int64', copy=False)

    @lazyval
    def asset_start_dates(self):
        return self.dates[self._country_group[LIFETIMES][START_DATE][:]]

    @lazyval
    def asset_end_dates(self):
        return self.dates[self._country_group[LIFETIMES][END_DATE][:]]

    @lazyval
    def _currency_codes(self):
        bytes_array = self._country_group[CURRENCY][CODE][:]
        return bytes_array_to_native_str_object_array(bytes_array)

    def currency_codes(self, sids):
        """Get currencies in which prices are quoted for the requested sids.

        Parameters
        ----------
        sids : np.array[int64]
            Array of sids for which currencies are needed.

        Returns
        -------
        currency_codes : np.array[object]
            Array of currency codes for listing currencies of ``sids``.
        """
        # Find the index of requested sids in our stored sids.
        ixs = self.sids.searchsorted(sids, side='left')

        result = self._currency_codes[ixs]

        # searchsorted returns the index of the next lowest sid if the lookup
        # fails. Fill these sids with the special "missing" sentinel.
        not_found = (self.sids[ixs] != sids)

        result[not_found] = None

        return result

    @property
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return pd.Timestamp(self.dates[-1], tz='UTC')

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        raise NotImplementedError(
            'HDF5 pricing does not yet support trading calendars.'
        )

    @property
    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return pd.Timestamp(self.dates[0], tz='UTC')

    @lazyval
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(self.dates, utc=True)

    def get_value(self, sid, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        """
        self._validate_assets([sid])
        self._validate_timestamp(dt)

        sid_ix = self.sids.searchsorted(sid)
        dt_ix = self.dates.searchsorted(dt.asm8)
        value = self._postprocessors[field](
            self._country_group[DATA][field][sid_ix + len(self.sids) * dt_ix]
        )

        # When the value is nan, this dt may be outside the asset's lifetime.
        # If that's the case, the proper NoDataOnDate exception is raised.
        # Otherwise (when there's just a hole in the middle of the data), the
        # nan is returned.
        if np.isnan(value):
            if dt.asm8 < self.asset_start_dates[sid_ix]:
                raise NoDataBeforeDate()

            if dt.asm8 > self.asset_end_dates[sid_ix]:
                raise NoDataAfterDate()

        return value


class MultiCountryDailyBarReader(CurrencyAwareSessionBarReader):
    """
    Parameters
    ---------
    readers : dict[str -> SessionBarReader]
        A dict mapping country codes to SessionBarReader instances to
        service each country.
    """

    def __init__(self, readers):
        self._readers = readers
        self._country_map = pd.concat([
            pd.Series(index=reader.sids, data=country_code)
            for country_code, reader in iteritems(readers)
        ])

    @classmethod
    def from_file(cls, h5_file):
        """
        Construct from an h5py.File.

        Parameters
        ----------
        h5_file : h5py.File
            An HDF5 daily pricing file.
        """
        return cls({
            country: HDFReader.from_file(h5_file, country)
            for country in h5_file.keys()
        })

    @classmethod
    def from_path(cls, path):
        """
        Construct from a file path.

        Parameters
        ----------
        path : str
            Path to an HDF5 daily pricing file.
        """
        return cls.from_file(h5py.File(path))

    @property
    def countries(self):
        """A set-like object of the country codes supplied by this reader.
        """
        return viewkeys(self._readers)

    def _country_code_for_assets(self, assets):
        country_codes = self._country_map.get(assets)

        # In some versions of pandas (observed in 0.22), Series.get()
        # returns None if none of the labels are in the index.
        if country_codes is not None:
            unique_country_codes = country_codes.dropna().unique()
            num_countries = len(unique_country_codes)
        else:
            num_countries = 0

        if num_countries == 0:
            raise ValueError('At least one valid asset id is required.')
        elif num_countries > 1:
            raise NotImplementedError(
                (
                    'Assets were requested from multiple countries ({}),'
                    ' but multi-country reads are not yet supported.'
                ).format(list(unique_country_codes))
            )

        return np.asscalar(unique_country_codes)

    def load_raw_arrays(self,
                        columns,
                        start_date,
                        end_date,
                        assets):
        """
        Parameters
        ----------
        columns : list of str
           'open', 'high', 'low', 'close', or 'volume'
        start_date: Timestamp
           Beginning of the window range.
        end_date: Timestamp
           End of the window range.
        assets : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        country_code = self._country_code_for_assets(assets)

        return self._readers[country_code].load_raw_arrays(
            columns,
            start_date,
            end_date,
            assets,
        )

    @property
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return max(
            reader.last_available_dt for reader in self._readers.values()
        )

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        raise NotImplementedError(
            'HDF5 pricing does not yet support trading calendars.'
        )

    @property
    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return min(
            reader.first_trading_day for reader in self._readers.values()
        )

    @property
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(
            reduce(
                np.union1d,
                (reader.dates for reader in self._readers.values()),
            ),
            utc=True,
        )

    def get_value(self, sid, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        NoDataForSid
            If the given sid is not valid.
        """
        try:
            country_code = self._country_code_for_assets([sid])
        except ValueError as exc:
            raise_from(
                NoDataForSid(
                    'Asset not contained in daily pricing file: {}'.format(sid)
                ),
                exc
            )
        return self._readers[country_code].get_value(sid, dt, field)

    def get_last_traded_dt(self, asset, dt):
        """
        Get the latest day on or before ``dt`` in which ``asset`` traded.

        If there are no trades on or before ``dt``, returns ``pd.NaT``.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset for which to get the last traded day.
        dt : pd.Timestamp
            The dt at which to start searching for the last traded day.

        Returns
        -------
        last_traded : pd.Timestamp
            The day of the last trade for the given asset, using the
            input dt as a vantage point.
        """
        country_code = self._country_code_for_assets([asset.sid])
        return self._readers[country_code].get_last_traded_dt(asset, dt)

    def currency_codes(self, sids):
        """Get currencies in which prices are quoted for the requested sids.

        Assumes that a sid's prices are always quoted in a single currency.

        Parameters
        ----------
        sids : np.array[int64]
            Array of sids for which currencies are needed.

        Returns
        -------
        currency_codes : np.array[S3]
            Array of currency codes for listing currencies of ``sids``.
        """
        country_code = self._country_code_for_assets(sids)
        return self._readers[country_code].currency_codes(sids)


def check_sids_arrays_match(left, right, message):
    """Check that two 1d arrays of sids are equal
    """
    if len(left) != len(right):
        raise ValueError(
            "{}:\nlen(left) ({}) != len(right) ({})".format(
                message, len(left), len(right)
            )
        )

    diff = (left != right)
    if diff.any():
        (bad_locs,) = np.where(diff)
        raise ValueError(
            "{}:\n Indices with differences: {}".format(message, bad_locs)
        )
