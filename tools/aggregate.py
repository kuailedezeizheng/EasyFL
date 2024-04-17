from defenses.fed_avg import federated_averaging
from defenses.flame import flame
from defenses.fltrust import fltrust
from defenses.krum import krum
from defenses.median import median
from defenses.multikrum import multikrum
from defenses.trimmed_mean import trimmed_mean


def get_aggregate_function(aggregate_function_name):
    """Define the aggregate function for training."""
    aggregate_functions = {
        'fed_avg': federated_averaging,
        'flame': flame,
        'median': median,
        'fltrust': fltrust,
        'trimmed_mean': trimmed_mean,
        'krum': krum,
        'multikrum': multikrum
    }
    if aggregate_function_name not in aggregate_functions:
        raise SystemExit("Error: unrecognized aggregate function!")

    func = aggregate_functions[aggregate_function_name]
    return func
