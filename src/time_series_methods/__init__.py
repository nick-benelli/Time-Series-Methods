def hello() -> str:
    return "Hello from time-series-methods!"


from . import accuracy, aggregator, dickey_fuller  # noqa: E402, F401
from .plot import ts_quick_plots  # noqa: E402, F401

from .avg_ts import average_time_series_by_county  # noqa: E402, F401
from .eacf import eacf  # noqa: E402, F401
from .shapiro_wilk import shapiro_wilk_test  # noqa: E402, F401
