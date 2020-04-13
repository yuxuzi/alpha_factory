"""
PipelineLoader accepting a DataFrame as input.
"""


from interface import implements

from zipline.pipeline.loaders.base import PipelineLoader
from pandas import HDFStore, DataFrame, Timestamp
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.testing.core import tmp_asset_finder
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.engine import SimplePipelineEngine
from alpha_factory.data.loaders.mydataset import MyDataSet
from zipline.pipeline import Pipeline
import pandas as pd
from zipline.pipeline.factors.factor import CustomFactor

class HDFLoader(implements(PipelineLoader)):
    """
    pipline loader from HDF file
    """

    def __init__(self, dataset, data_dir, ):  # start_date, end_date
        with HDFStore(data_dir) as store:
            loaders = {}
            for column in dataset.columns:
                dat = store.select(column.name).astype(
                    column.dtype).unstack().tz_localize('utc')
                dat.columns=list(range(len(dat.columns)))
                   # where="date >='{}' and date <'{}'".format(start_date,end_date

                loaders[column] = DataFrameLoader(
                    column=column,
                    baseline=dat,
                    adjustments=None,
                )

            self._loaders = loaders

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        """
        Load by delegating to sub-loaders.
        """
        out = {}
        for col in columns:
            try:
                loader = self._loaders.get(col)
                if loader is None:
                    loader = self._loaders[col.unspecialize()]
            except KeyError:
                raise ValueError("Couldn't find loader for %s" % col)
            out.update(
                loader.load_adjusted_array(domain, [col], dates, sids, mask)
            )
        return out


def make_asset_finder(start_date, end_date):
    num_assets=9
    equities = DataFrame({
        'sid': list(range(num_assets)),
        'start_date': [start_date] * num_assets,
        'end_date': [end_date] * num_assets,
        'symbol': ['MOD', 'MODN', 'MOFG', 'MOG_A', 'MOH', 'MON', 'MORN', 'MOS', 'MOSY'],
        'exchange': 'NYSE',
        'country_code':'US'
    })

    exchanges=DataFrame({'exchange': ['NYSE'], 'country_code': ['US']})


    return tmp_asset_finder(equities=equities,exchanges=exchanges ).__enter__()


def make_simple_pipeline_Engline(start_date, end_date):
    data_dir = "/home/yuxuzi/Data/mydataset"
    calendar = US_EQUITIES
    loader = HDFLoader(MyDataSet, data_dir)
    finder = make_asset_finder(start_date, end_date)
    return SimplePipelineEngine(lambda col:loader, finder, default_domain=calendar)


class RollingSumDifference(CustomFactor):
    window_length = 3
    inputs = [MyDataSet.open, MyDataSet.close]

    def compute(self, today, assets, out, open, close):
        out[:] = (open - close).sum(axis=0)

if __name__ == '__main__':
    start_date, end_date=Timestamp('2018-03-12'),Timestamp('2018-03-27')
    engine = make_simple_pipeline_Engline(start_date, end_date)
    pipe = Pipeline(columns={'close': MyDataSet.close.latest,
                             'sumdiff': RollingSumDifference()
                             }, )
    df = engine.run_pipeline(pipe,start_date, end_date)
    print(df)
