# data-analytics

Small utilities for exploratory work with **pandas** (descriptive statistics).

## Install

```bash
uv add "data-analytics-mcuig==0.1.0" --extra-index-url https://test.pypi.org/simple/  --index-strategy unsafe-best-match

```


## Quick example




```python
import pandas as pd
from data_analytics_mcuig.column_statistics import calculate_min_of_column

df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
min_a = calculate_min_of_column(df, "a")  # 1.0
```


## Documentation

- **[API reference](api.md)** — generated from docstrings.

## Source

[github.com/ldirer/pycon-workshop-data-analytics](https://github.com/ldirer/pycon-workshop-data-analytics)
