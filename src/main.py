import estimators
import itertools
import pandas as pd
import dgp
import os

if __name__ == "__main__":
    output_dir = "../output/"
    dgp_params = {
        "confounding_strength": 0.2,
        "pC": 0.5,
        "po_Y_1": 0.024,
        "po_Y_0": 0.03,
        "pA1_C0": 0.2,
        "pA1_C1": 0.3,
    }

    variable_params = dict(error_rate=[0.01, 0.02, 0.05], recall_rate=[0.7, 0.8, 0.9])

    fixed_params = dict(bootstraps=5, sample_size=100)

    graph = dgp.create_twin_error_graph()
    cpds = dgp.create_statins_stroke_cpds(**dgp_params)

    true_or = estimators.compute_or(dgp_params["po_Y_0"], dgp_params["po_Y_1"])

    all_params = list(
        dict(zip(list(variable_params), e))
        for e in itertools.product(*list(variable_params.values()))
    )

    results = []
    for param in all_params:

        model = dgp.create_twin_error_model(graph, cpds, error_rate=param["error_rate"])

        for i in range(fixed_params["bootstraps"]):

            df = model.simulate(
                n_samples=int(param["recall_rate"] * fixed_params["sample_size"])
            )

            y0, y1 = estimators.ipw(df, treatment="A", outcome="Y", confounders="C")

            odds_ratio = estimators.compute_or(y0, y1)

            results.append(
                fixed_params
                | param
                | {"est_odds_ratio": odds_ratio, "est_y0": y0, "est_y1": y1}
            )

    final_results = pd.DataFrame(results)

    output = os.path.join(
        output_dir, f"b{fixed_params['bootstraps']}_n{fixed_params['sample_size']}.csv"
    )
    final_results.to_csv(output)
