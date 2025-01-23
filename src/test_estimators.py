import pandas as pd
import numpy as np
import pytest

import estimators


def test_ipw():

    df = pd.DataFrame({"A": [0, 0, 0, 1, 1, 1], "C": [0, 1, 1, 0, 0, 1], "Y": [1, 0, 0, 0, 1, 1]})
    y0, y1 = estimators.ipw(df, "A", "Y", "C")
    assert pytest.approx(0.5) == y0
    assert pytest.approx(0.75) == y1


