from types import SimpleNamespace
import copy
from config_helper import check_and_get_configuration

# Function to recursively convert dictionaries and lists into SimpleNamespace objects
def __dict_to_simplenamespace(d):
    if isinstance(d, dict):
        # Convert each key-value pair in the dictionary
        for k, v in d.items():
            d[k] = __dict_to_simplenamespace(v)
        # Convert the dictionary itself into a SimpleNamespace
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        # If the object is a list, convert each element in the list
        return [__dict_to_simplenamespace(i) for i in d]
    else:
        # If the object is neither a dictionary nor a list, return it as is
        return d

# Function to generate all combinations of hyperparameter configurations
def generate_configs(cfg_obj):
    # Convert the input configuration object into SimpleNamespace format
    config = __dict_to_simplenamespace(cfg_obj)
    
    # Extract the hyperparameters from the configuration
    hyper_params = config.hyper_parameters
    
    # Identify the hyperparameters that need to be optimized
    params_to_optimize = {k: v for k, v in vars(hyper_params).items() if isinstance(v, SimpleNamespace) and v.optimize}

    # Generate ranges for each hyperparameter that needs to be optimized
    param_ranges = {k: range(v.start, v.end + 1, v.step) for k, v in params_to_optimize.items()}

    # Recursive function to generate all combinations of hyperparameter values
    def generate_combinations(params):
        if not params:
            # Base case: if no more parameters, yield an empty dictionary
            yield {}
            return
        # Pop an item from the parameter dictionary
        param, values = params.popitem()
        for value in values:
            # Recursively generate combinations for the remaining parameters
            for combination in generate_combinations(params.copy()):
                combination[param] = value
                yield combination

    # Generate all possible combinations of hyperparameters
    combinations = list(generate_combinations(param_ranges.copy()))

    # Create a new configuration object for each combination of hyperparameters
    config_list = []
    for combination in combinations:
        # Deep copy the original configuration to create a new one
        new_config = copy.deepcopy(config)
        for param, value in combination.items():
            # Update the hyperparameter values in the new configuration
            setattr(new_config.hyper_parameters, param, SimpleNamespace(**{**vars(getattr(new_config.hyper_parameters, param)), "value": value}))
        config_list.append(new_config)

    return config_list

# Main script execution
if __name__ == '__main__':    
    # Load the configuration object and schema from JSON files
    cfg_obj = check_and_get_configuration('./config/config.json', './config/config_schema.json')
    
    # Generate all possible configurations based on the hyperparameter combinations
    configs = generate_configs(cfg_obj)
    
    # Print or use the generated configurations
    for i, conf in enumerate(configs):
        print(f"Config {i + 1}:\n", conf)
