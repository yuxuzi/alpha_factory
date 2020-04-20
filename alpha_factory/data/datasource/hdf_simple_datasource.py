import timeit
import numpy as np
from interface import implements
import pandas as pd
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.testing.core import tmp_asset_finder
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.engine import SimplePipelineEngine

from zipline.pipeline import Pipeline
from zipline.pipeline.factors.factor import CustomFactor
from zipline.lib.adjusted_array import AdjustedArray
from zipline.utils.numpy_utils import as_column
from zipline.pipeline.data.dataset import Column, DataSet
from alpha_factory.data.datasource.mydataset import MyDataSet


def _validate_input_column(self, column):
    """Make sure a passed column is our column.
    """
    if column != self.column and column.unspecialize() != self.column:
        raise ValueError("Can't load unknown column %s" % column)


class HDFSimpleLoader(implements(PipelineLoader)):
    """
    pipline loader from HDF file
    """

    def __init__(self, data_path, dates, assets):  # start_date, end_date
        self.data_path = data_path
        self.dates = dates.values
        self.assets = assets.values


    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        date_indexer =np.searchsorted(self.dates, dates)
        assets_indexer = np.searchsorted(self.assets, sids)

        # Boolean arrays with True on matched entries
        good_dates = (date_indexer != -1)
        good_assets = (assets_indexer != -1)
        mask = (good_assets & as_column(good_dates)) & mask
        out = {}
        with pd.HDFStore(self.data_path) as store:
            for column in columns:
                try:
                    data = store["/data/"+column.name].values
                    data = data[np.ix_(date_indexer, assets_indexer)]
                    data[~mask] = column.missing_value
                except KeyError:
                    raise ValueError("Couldn't find loader for %s" % column.name)
                out[column] = AdjustedArray(
                    # Pull out requested columns/rows from our baseline data.
                    data=data,
                    adjustments={},
                    missing_value=column.missing_value,
                )
        return out


class HDFSimpleDataSource:
    def __init__(self, data_path, dataset=None):
        self.data_path = data_path
        with pd.HDFStore(data_path) as store:
            self.sids=store["index/sids"]
            self.dates=store["index/dates"]
            equities=store['equities'].assign(
                        exchange='NYSE',
                        country_code='US'
            )

        self.loader = HDFSimpleLoader(data_path, self.dates, self.sids)
        exchanges = pd.DataFrame({'exchange': ['NYSE'], 'country_code': ['US']})
        self.asset_finder = tmp_asset_finder(url='sqlite:///:memory:', equities=equities,
                                             exchanges=exchanges).__enter__()

    def write_hdf(data, data_path):
        try:
            data.index = data.index.set_levels(data.index.levels[0].tz_localize('utc'), level=0)
        except:
            pass

        symbols = data.index.levels[1].unique()
        num_assets = len(symbols)
        sids = pd.Series(np.arange(num_assets))
        dates = pd.Series(data.index.levels[0].unique())
        import pdb;
        pdb.set_trace()
        dtypes = pd.Series(data.dtypes)
        starts, ends = [], []
        with pd.HDFStore(data_path, complevel=9, mode='w') as store:
            for column in data:
                dat = data[column].unstack().set_axis(sids, axis=1, inplace=False)
                dat_is_null = dat.isnull()
                starts.append(dat_is_null.apply(lambda x: x.idxmin()))
                ends.append(dat_is_null.apply(lambda x: x[::-1].idxmin()))
                store.put('/data/' + column, dat)
            start_dates = np.amin(np.vstack(starts), axis=0)
            end_dates = np.amax(np.vstack(ends), axis=0)

            equities = pd.DataFrame({'sid': sids,
                                     'start_date': start_dates,
                                     'end_date': end_dates,
                                     'symbol': symbols}
                                    )

            store.put('/index/dates', dates)
            store.put('/index/sids', sids)
            store.put('/dtypes', dtypes)
            store.put('/equities', equities)

    def infer_dataset(self, fields=None):
        with pd.HDFStore(self.data_path) as store:
            dtypes = store['/dtypes']
            columns = {k: Column(v) for k, v in dtypes.iteritems() if fields is None or k in fields}
        return type("Dataset", (DataSet,), columns)

    def run_pipeline(self, *args, **kwargs):
        calendar = US_EQUITIES
        loader = self.loader
        finder = self.asset_finder
        engine = SimplePipelineEngine(lambda col: loader, finder, default_domain=calendar)
        return engine.run_pipeline(*args, **kwargs)


class RollingSumDifference(CustomFactor):
    window_length = 3
    inputs = [MyDataSet.open, MyDataSet.close]

    def compute(self, today, assets, out, open, close):
        out[:] = (open - close).sum(axis=0)


if __name__ == '__main__':
    def cal():
        start_date, end_date = pd.Timestamp('2018-03-12'), pd.Timestamp('2018-03-27')
        data_source = HDFSimpleDataSource("/home/yuxuzi/Data/mydataset", MyDataSet)
        pipe = Pipeline(columns={'close': MyDataSet.close.latest,
                                 'sumdiff': RollingSumDifference()
                                 }, )
        df = data_source.run_pipeline(pipe, start_date, end_date)
        print(df)


    t1 = timeit.default_timer()
    timeit.timeit('cal()', number=10, setup="from __main__ import cal")
    print("{} Seconds needed for single threaded execution".format(timeit.default_timer() - t1))
