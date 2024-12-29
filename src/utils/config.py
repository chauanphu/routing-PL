import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# PSO hyperparameters
POPULATION_SIZE = config['PSO']['population_size']
MAX_ITER = config['PSO']['max_iter']
COGNITIVE_WEIGHT = config['PSO']['cognitive_weight']
SOCIAL_WEIGHT = config['PSO']['social_weight']
INFEASIBILITY_PENALTY = config['PSO']['infeasibility_penalty']
INERTIA = config['PSO']['inertia']

# Constraints
ALLOW_EARLY = config['constraints']['allow_early']