from alfred.models.stockformer.stockformer import *
from .lstm import *
from .advanced_lstm import *
from .linear import *
from .trans_am import *

model_config = {
    'lstm': LSTMModel,
    'lstm-conv1d': LSTMConv1d,
    'advanced-lstm': AdvancedLSTM,
    'trans-am': TransAm
}