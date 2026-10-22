"""Deployable causal v3 bundle; predictions never receive truth labels."""
from core import validate_trajectory,score_anomaly

def predict_bundle(bundle,raw):
    validate_trajectory(raw)
    features=bundle['preprocessor'].transform(raw)
    out=features[['engine_id','cycle']].copy()
    out['predicted_rul']=bundle['estimator'].predict(features)
    out['warning_probability']=bundle['warning_classifier'].predict_probability(features)
    out['warning_threshold']=float(bundle['warning_threshold'])
    out['warning_horizon']=int(bundle['warning_horizon'])
    out['mad_score']=score_anomaly(bundle['anomaly'],features)
    out['mad_threshold']=float(bundle['anomaly']['threshold'])
    return out
