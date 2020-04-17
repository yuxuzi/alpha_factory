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
from zipline.assets import AssetFinder
from zipline.pipeline.factors.factor import CustomFactor

class HDFSimpleDataSource:
    def __init__(self, dataset,  data_path):
        with HDFStore(data_path) as store:
            self.dates=dates
            self.symbols=symbols
            self.num_assets=len(symbols)
        self.dataset=dataset
        self.loader=HDFSimpleLoader(dataset, data_path)
    def get_asset_finder(self):
        equities = DataFrame({
            'sid': list(range(self.num_assets)),
            'start_date': [start_date] * self.num_assets,
            'end_date': [end_date] * self.num_assets,
            'symbol': self.symbols,
            'exchange': 'NYSE',
            'country_code': 'US'
        })

        exchanges = DataFrame({'exchange': ['NYSE'], 'country_code': ['US']})
        return AssetFinder(url='sqlite:///:memory:',equities=equities, exchanges=exchanges)

    def run_pipeline(self, start_date, end_date, *args, **kwargs):
        calendar = US_EQUITIES
        loader = self.loader
        finder = self.get_asset_finder(start_date, end_date)
        engine=SimplePipelineEngine(lambda col: loader, finder, default_domain=calendar)
        return engine.run_pipeline(*args, **kwargs)



class HDFSimpleLoader(implements(PipelineLoader)):
    """
    pipline loader from HDF file
    """

    def __init__(self, dataset, data_path ):  # start_date, end_date

        with HDFStore(data_path) as store:
            loaders = {}
            for column in dataset.columns:
                dat = store.select(column.name).astype(
                    column.dtype).unstack().tz_localize('utc')
                dat.columns=list(range(len(dat.columns)))
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


class RollingSumDifference(CustomFactor):
    window_length = 3
    inputs = [MyDataSet.open, MyDataSet.close]

    def compute(self, today, assets, out, open, close):
        out[:] = (open - close).sum(axis=0)

if __name__ == '__main__':
    start_date, end_date=Timestamp('2018-03-12'),Timestamp('2018-03-27')
    data_source=HDFSimpleDataSource()

    pipe = Pipeline(columns={'close': MyDataSet.close.latest,
                             'sumdiff': RollingSumDifference()
                             }, )
    df = data_source.run_pipeline(pipe,start_date, end_date)
    print(df)
