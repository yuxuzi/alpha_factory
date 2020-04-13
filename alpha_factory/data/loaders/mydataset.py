from zipline.utils.numpy_utils import float64_dtype, categorical_dtype

from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.data.dataset import Column, DataSet

MyDataSet = type("MyDataset",
                 (DataSet,),
                 {x: Column(float64_dtype) for x in ['open',
                                                     'high',
                                                     'low',
                                                     'close',
                                                     'volume',
                                                     # 'ex_dividend',
                                                     # 'split_ratio',
                                                     # 'adj_open',
                                                     # 'adj_high',
                                                     # 'adj_low',
                                                     # 'adj_close',
                                                     # 'adj_volume'
                 ]}
                 )

if __name__ == '__main__':
    dat = MyDataSet

