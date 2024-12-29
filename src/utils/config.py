import yaml

# Load config file
with open('param.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# PSO hyperparameters
POPULATION_SIZE = config['PSO']['population_size']
MAX_ITER = config['PSO']['max_iter']
COGNITIVE_WEIGHT = config['PSO']['cognitive_weight']
SOCIAL_WEIGHT = config['PSO']['social_weight']
INFEASIBILITY_PENALTY = config['constraints']['infeasbiility_penalty']
INERTIA = config['PSO']['inertia_weight']

# GA hyperparameters
GA_NUMMAX = config['GA']['num_max']
GA_NUMMIN = config['GA']['num_min']
GA_PSMIN = config['GA']['ps_min']
GA_PSMAX = config['GA']['ps_max']
GAMMA = config['GA']['gamma']
BETA = config['GA']['beta']
GA_MINITER = config['GA']['min_iter']
GA_MAXITR = config['GA']['max_iter']

# Constraints
ALLOW_EARLY = config['constraints']['allow_early']