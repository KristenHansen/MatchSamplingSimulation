import numpy as np
from statsmodels.discrete.discrete_model import Logit
from statsmodels.treatment.treatment_effects import TreatmentEffect


def ipw(df, treatment, outcome, confounders):

    treatment_model = Logit.from_formula(f'{treatment} ~ {"+".join(confounders)}', df).fit()
    propensity_score = treatment_model.predict()
    weight = np.where(propensity_score == 1, 1 / propensity_score, 1 / (1 - propensity_score))

    y1 = np.sum(df.loc[df[treatment] == 1, outcome] * weight[df[treatment] == 1]) / np.sum(df[treatment] == 1)
    y0 = np.sum(df.loc[df[treatment] == 0, outcome] * weight[df[treatment] == 0]) / np.sum(df[treatment] == 0)

    return y0, y1

def compute_or(y0, y1):
    return y1 * (1 - y0) / ((1 - y1) * y0)


