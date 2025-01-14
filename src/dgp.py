import pgmpy
from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


def create_twin_error_graph() -> list:
    """
    Creates the twin-error graph model for a conditionally ignorable model with errors.
    The errors are nondifferential, and structured such that whenever one occurs, the treatment A
    and outcome Y are replaced by another distinct draw from the same structural equations (i.e.
    corresponds to data collected from a linked source, in which linkage errors could occur). It
    is assumed that the baseline covariate C is always accurate (i.e. corresponds to data that
    is already in possession of the analyst, and is considered correct).

    """


    edges = [
        ("C", "A"),
        ("C", "Y"),
        ("A", "Y"),
        ("C1", "A1"),
        ("C1", "Y1"),
        ("A1", "Y1"),
        ("C1", "C_obs"),
        ("C", "C_obs"),
        ("A", "A_obs"),
        ("A1", "A_obs"),
        ("Y", "Y_obs"),
        ("Y1", "Y_obs"),
        ("E", "A_obs"),
        ("E", "C_obs"),
        ("E", "Y_obs"),
    ]


    return edges


def create_statins_stroke_cpds(confounding_strength):
    """
    Creates simulation parameters based on the effect of statins on the reduction of ischemic stroke
    risk in Type 2 diabetes (https://pubmed.ncbi.nlm.nih.gov/20979581/)

    A: statin usage
    Y: stroke status
    C: sex

    Causal effect: OR = 0.79
    where OR = (p(Y(a=1)=1) / (1 - p(Y(a=1)=1))) / (p(Y(a=0)=1) / (1 - p(Y(a=0) = 1)))


    Confounding is introduced in the simulation by adjusting the strength of the confounding
    parameter.
    """

    # Parameters derived from table 2
    # Note that this results in an OR of 0.8
    po_Y_1 = 0.024  # p(Y(a=1) = 1)
    po_Y_0 = 0.03  # p(Y(a=0) = 1)
    pC = 0.5  # p(C = 0)

    # Create p(A | C), which can be arbitrary since this does not contribute to the causal effect
    pA1_C0 = 0.2
    pA1_C1 = 0.3

    # Create p(Y | A, C) based on a deviation from the population potential outcomes and the
    # confounding strength

    pY1_A1C0 = po_Y_1 * (1 + confounding_strength / pC)
    pY1_A1C1 = po_Y_1 * (1 - confounding_strength / (1 - pC))
    pY1_A0C0 = po_Y_0 * (1 + confounding_strength / pC)
    pY1_A0C1 = po_Y_0 * (1 - confounding_strength / (1 - pC))

    # Convert into CPDs

    cpd_c = TabularCPD(variable="C", variable_card=2, values=[[pC], [1 - pC]])

    cpd_a = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[1 - pA1_C0, 1 - pA1_C1], [pA1_C0, pA1_C1]],
        evidence=["C"],
        evidence_card=[2],
    )

    cpd_y = TabularCPD(
        variable="Y",
        variable_card=2,
        values=[
            [1 - pY1_A0C0, 1 - pY1_A0C1, 1 - pY1_A1C0, 1 - pY1_A1C1],
            [pY1_A0C0, pY1_A0C1, pY1_A1C0, pY1_A1C1],
        ],
        evidence=["A", "C"],
        evidence_card=[2, 2],
    )

    cpds = {"A": cpd_a, "C": cpd_c, "Y": cpd_y}

    return cpds


def create_twin_error_model(graph: list, cpds: dict, error_rate: float):
    """
    Creates the twin-error model over binary variables. Only CPDs relevant to the conditionally ignorable model need to be
    supplied.

    Parameters
    ----------
    graph : BayesianNetwork

    cpds : dict

    error_rate : float
        The rate of match sampling errors


    """
    cpd_c1 = TabularCPD(variable="C1", variable_card=2, values=cpds["C"].get_values())

    cpd_a1 = TabularCPD(
        variable="A1",
        variable_card=2,
        values=cpds["A"].get_values(),
        evidence=["C1"],
        evidence_card=[2],
    )

    cpd_y1 = TabularCPD(
        variable="Y1",
        variable_card=2,
        values=cpds["Y"].get_values(),
        evidence=['A1', "C1"],
        evidence_card=[2, 2],
    )


    # error_rate = p(E = 1)
    cpd_e = TabularCPD(
        variable="E", variable_card=2, values=[[1 - error_rate], [error_rate]]
    )

    cpd_c_obs = TabularCPD(
        variable="C_obs",
        variable_card=2,
        values=[[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]],
        evidence=["E", "C1", "C"],
        evidence_card=[2, 2, 2],
    )
    cpd_y_obs = TabularCPD(
        variable="Y_obs",
        variable_card=2,
        values=[[1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0, 1, 1]],
        evidence=["E", "Y1", "Y"],
        evidence_card=[2, 2, 2],
    )

    cpd_a_obs = TabularCPD(
        variable="A_obs",
        variable_card=2,
        values=[[1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0, 1, 1]],
        evidence=["E", "A1", "A"],
        evidence_card=[2, 2, 2],
    )

    all_cpds = cpds | {
        "C1": cpd_c1,
        "A1": cpd_a1,
        "Y1": cpd_y1,
        "E": cpd_e,
        "A_obs": cpd_a_obs,
        "C_obs": cpd_c_obs,
        "Y_obs": cpd_y_obs,
    }

    model = BayesianNetwork()
    model.add_edges_from(graph)
    model.add_cpds(*list(all_cpds.values()))

    return model
