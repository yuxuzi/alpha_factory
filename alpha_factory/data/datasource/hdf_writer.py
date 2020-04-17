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


def days_and_sids_for_fields(fields):
    """
    Returns the date index and sid columns shared by a list of dataframes,
    ensuring they all match.

    Parameters
    ----------
    frames : list[pd.DataFrame]
        A list of dataframes indexed by day, with a column per sid.

    Returns
    -------
    days : np.array[datetime64[ns]]
        The days in these dataframes.
    sids : np.array[int64]
        The sids in these dataframes.

    Raises
    ------
    ValueError
        If the dataframes passed are not all indexed by the same days
        and sids.
    """
    if not fields:
        days = np.array([], dtype='datetime64[ns]')
        sids = np.array([], dtype='int64')
        return days, sids

    # Ensure the indices and columns all match.
    frames = [field.unstack() for field in fields]
    check_indexes_all_same(
        [frame.index for frame in frames],
        message='Frames have mismatched days.',
    )
    check_indexes_all_same(
        [frame.columns for frame in frames],
        message='Frames have mismatched sids.',
    )

    return frames[0].index.values, frames[0].columns.values


class HDFWriter(object):
    """
    Class capable of writing daily OHLCV data to disk in a format that
    can be read efficiently by HDF5DailyBarReader.

    Parameters
    ----------
    filename : str
        The location at which we should write our output.
    date_chunk_size : int
        The number of days per chunk in the HDF5 file. If this is
        greater than the number of days in the data, the chunksize will
        match the actual number of days.

    See Also
    --------
    zipline.data.hdf5_daily_bars.HDF5DailyBarReader
    """

    def __init__(self, filename, date_chunk_size, dataset=None):
        self._filename = filename
        self._date_chunk_size = date_chunk_size
        self.fields = [Field(x.name, x.dtype) for x in dataset.columns]

    def h5_file(self, mode):
        return h5py.File(self._filename, mode)

    def write(self,
              country_code,
              frames,
              currency_codes=None,
              scaling_factors=None):
        """
        Write the OHLCV data for one country to the HDF5 file.

        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code for this country.
        frames : dict[str, pd.DataFrame]
            A dict mapping each OHLCV field to a dataframe with a row
            for each date and a column for each sid. The dataframes need
            to have the same index and columns.
        currency_codes : pd.Series, optional
            Series mapping sids to 3-digit currency code values for those sids'
            listing currencies. If not passed, missing currencies will be
            written.
        scaling_factors : dict[str, float], optional
            A dict mapping each OHLCV field to a scaling factor, which
            is applied (as a multiplier) to the values of field to
            efficiently store them as uint32, while maintaining desired
            precision. These factors are written to the file as metadata,
            which is consumed by the reader to adjust back to the original
            float values. Default is None, in which case
            DEFAULT_SCALING_FACTORS is used.
        """
        # if scaling_factors is None:
        #     scaling_factors = DEFAULT_SCALING_FACTORS

        # Note that this functions validates that all of the frames
        # share the same days and sids.
        days, sids = days_and_sids_for_fields(list(frames.values()))

        # XXX: We should make this required once we're using it everywhere.
        if scaling_factors is None:
            scaling_factors = {x.name: 1 for x in self.fields}
        if currency_codes is None:
            currency_codes = pd.Series(index=sids, data=MISSING_CURRENCY)

        # Currency codes should match dataframe columns.
        check_sids_arrays_match(
            sids,
            currency_codes.index.values,
            message="currency_codes sids do not match data sids:",
        )

        # Write start and end dates for each sid.
        start_date_ixs, end_date_ixs = compute_asset_lifetimes(frames)

        if len(sids):
            chunks = (min(self._date_chunk_size, len(days)) * len(sids),)  # check
        else:
            # h5py crashes if we provide chunks for empty data.
            chunks = None

        with self.h5_file(mode='a') as h5_file:
            # ensure that the file version has been written
            h5_file.attrs['version'] = VERSION

            country_group = h5_file.create_group(country_code)

            self._write_index_group(country_group, days, sids)
            self._write_lifetimes_group(
                country_group,
                start_date_ixs,
                end_date_ixs,
            )
            self._write_currency_group(country_group, currency_codes)
            self._write_data_group(
                country_group,
                frames,
                scaling_factors,
                chunks,
            )

    def write_from_sid_df_pairs(self,
                                country_code,
                                data,
                                currency_codes=None,
                                scaling_factors=None):
        """
        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code for this country.
        data : iterable[tuple[int, pandas.DataFrame]]
            The data chunks to write. Each chunk should be a tuple of
            sid and the data for that asset.
        currency_codes : pd.Series, optional
            Series mapping sids to 3-digit currency code values for those sids'
            listing currencies. If not passed, missing currencies will be
            written.
        scaling_factors : dict[str, float], optional
            A dict mapping each OHLCV field to a scaling factor, which
            is applied (as a multiplier) to the values of field to
            efficiently store them as uint32, while maintaining desired
            precision. These factors are written to the file as metadata,
            which is consumed by the reader to adjust back to the original
            float values. Default is None, in which case
            DEFAULT_SCALING_FACTORS is used.
        """
        data = list(data)
        if not data:
            empty_frame = pd.DataFrame(
                index=[np.array([], dtype='datetime64[ns]'), np.array([], dtype='int64')]
            )
            return self.write(
                country_code,
                {f: empty_frame.copy() for f in self.fields},
                scaling_factors,
            )

        sids, frames = zip(*data)

        data = pd.concat(frames, keys=sids).rename_axis(['sid', 'date']).reset_index().set_index(['date', 'sid'])

        frames = {field.name: data[field] for field in self.fields}

        return self.write(
            country_code=country_code,
            frames=frames,
            scaling_factors=scaling_factors,
            currency_codes=currency_codes
        )

    def _write_index_group(self, country_group, days, sids):
        """Write /country/index.
        """
        index_group = country_group.create_group(INDEX)
        self._log_writing_dataset(index_group)

        index_group.create_dataset(SID, data=sids)

        # h5py does not support datetimes, so they need to be stored
        # as integers.
        index_group.create_dataset(DAY, data=days.astype(np.int64))

    def _write_lifetimes_group(self,
                               country_group,
                               start_date_ixs,
                               end_date_ixs):
        """Write /country/lifetimes
        """
        lifetimes_group = country_group.create_group(LIFETIMES)
        self._log_writing_dataset(lifetimes_group)

        lifetimes_group.create_dataset(START_DATE, data=start_date_ixs)
        lifetimes_group.create_dataset(END_DATE, data=end_date_ixs)

    def _write_currency_group(self, country_group, currencies):
        """Write /country/currency
        """
        currency_group = country_group.create_group(CURRENCY)
        self._log_writing_dataset(currency_group)

        currency_group.create_dataset(
            CODE,
            data=currencies.values.astype(dtype='S3'),
        )

    def _write_data_group(self,
                          country_group,
                          frames,
                          scaling_factors,
                          chunks):
        """Write /country/data
        """
        data_group = country_group.create_group(DATA)
        self._log_writing_dataset(data_group)

        for field in self.fields:
            data = frames[field.name].sort_index().astype('float64')  # may change type later

            dataset = data_group.create_dataset(
                field,
                compression='lzf',
                shuffle=True,
                data=data,
                chunks=chunks,
            )
            self._log_writing_dataset(dataset)

            dataset.attrs['scaling_factor'] = scaling_factors[field.name]

            log.debug(
                'Writing dataset {} to file {}',
                dataset.name, self._filename
            )

    def _log_writing_dataset(self, dataset):
        log.debug("Writing {} to file {}", dataset.name, self._filename)


def compute_asset_lifetimes(frames, fields):
    """
    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        A dict mapping each OHLCV field to a dataframe with a row for
        each date and a column for each sid, as passed to write().

    Returns
    -------
    start_date_ixs : np.array[int64]
        The index of the first date with non-nan values, for each sid.
    end_date_ixs : np.array[int64]
        The index of the last date with non-nan values, for each sid.
    """
    # Build a 2D array (dates x sids), where an entry is True if all
    # fields are nan for the given day and sid.
    is_null_matrix = np.logical_and.reduce(
        [frames[field].unstack().isnull().values for field in fields],
    )
    if not is_null_matrix.size:
        empty = np.array([], dtype='int64')
        return empty, empty.copy()

    # Offset of the first null from the start of the input.
    start_date_ixs = is_null_matrix.argmin(axis=0)
    # Offset of the last null from the **end** of the input.
    end_offsets = is_null_matrix[::-1].argmin(axis=0)
    # Offset of the last null from the start of the input
    end_date_ixs = is_null_matrix.shape[0] - end_offsets - 1

    return start_date_ixs, end_date_ixs





