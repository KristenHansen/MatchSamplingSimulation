import pytest
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from numpy.testing import assert_almost_equal

import dgp


def test_model():
    graph = dgp.create_twin_error_graph()
    cpds = dgp.create_statins_stroke_cpds(confounding_strength=.2)
    model = dgp.create_twin_error_model(graph, cpds, error_rate=.02)

    df = model.simulate(n_samples=1000)

def test_model_distribution_equals_original_under_no_error():
    graph = dgp.create_twin_error_graph()
    cpds = dgp.create_statins_stroke_cpds(confounding_strength=.2)

    original_model = BayesianNetwork()
    original_model.add_edges_from([('C', 'A'), ('A', 'Y'), ('C', 'Y')])
    original_model.add_cpds(*list(cpds.values()))


    model = dgp.create_twin_error_model(graph, cpds, error_rate=0)


    inference = VariableElimination(model)
    dist = inference.query(['A_obs', 'C_obs', 'Y_obs'])
    original_inference = VariableElimination(model)
    original_dist = original_inference.query(['A', 'C', 'Y'])

    assert_almost_equal(dist.values, original_dist.values)


