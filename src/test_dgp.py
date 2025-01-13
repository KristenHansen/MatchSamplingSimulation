import dgp

import pytest

def test_model():
    graph = dgp.create_twin_error_graph()
    cpds = dgp.create_statins_stroke_cpds(confounding_strength=.2)
    model = dgp.create_twin_error_model(graph, cpds, error_rate=.02)

    df = model.simulate(n_samples=1000)



