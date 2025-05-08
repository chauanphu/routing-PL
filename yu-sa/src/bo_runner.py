import sys
import json
from skopt.space import Integer, Real
from skopt import Optimizer

def load_paramspace(param_file):
    with open(param_file, 'r') as f:
        paramspace = json.load(f)
    search_space = []
    param_names = []
    for k, v in paramspace.items():
        if not v.get("tune", True):
            continue
        rng = v.get("range", [])
        if not rng or (len(rng) == 2 and rng[0] == rng[1]):
            continue  # skip fixed or empty range
        if v["type"] == "int":
            search_space.append(Integer(rng[0], rng[1], name=k))
        elif v["type"] == "float":
            search_space.append(Real(rng[0], rng[1], name=k))
        else:
            continue
        param_names.append(k)
    return search_space, param_names, paramspace

def load_history(hist_file, param_names):
    with open(hist_file, 'r') as f:
        history = json.load(f)
    X = []
    y = []
    for entry in history:
        x = []
        for k in param_names:
            x.append(entry.get(k))
        X.append(x)
        y.append(entry["score"])
    return X, y

def main():
    if len(sys.argv) != 3:
        print("Usage: bo_runner.py <paramspace.json> <history.json>", file=sys.stderr)
        sys.exit(1)
    try:
        param_file = sys.argv[1]
        hist_file = sys.argv[2]
        search_space, param_names, paramspace = load_paramspace(param_file)
        # If no tunable parameters, just return defaults
        if not search_space:
            next_config = {k: v["default"] for k, v in paramspace.items()}
            print(json.dumps(next_config))
            return
        X, y = load_history(hist_file, param_names)
        opt = Optimizer(
            dimensions=search_space,
            base_estimator="GP",
            acq_func="EI",
            acq_optimizer="auto",
            random_state=42
        )
        if len(X) > 0:
            opt.tell(X, y)
        next_x = opt.ask()
        next_config = {}
        for i, k in enumerate(param_names):
            if paramspace[k]["type"] == "int":
                next_config[k] = int(round(next_x[i]))
            else:
                next_config[k] = float(next_x[i])
        # Add untuned params with default values
        for k, v in paramspace.items():
            if k not in next_config:
                next_config[k] = v["default"]
        print(json.dumps(next_config))
    except Exception as e:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()