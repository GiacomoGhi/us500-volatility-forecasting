from types import SimpleNamespace
import copy

from config_helper import check_and_get_configuration


def __dict_to_simplenamespace(d):
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = __dict_to_simplenamespace(v)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [__dict_to_simplenamespace(i) for i in d]
    else:
        return d

def generate_configs(config):
    config = __dict_to_simplenamespace(cfg_obj)
    
    hyper_params = config.hyper_parameters
    
    params_to_optimize = {k: v for k, v in vars(hyper_params).items() if isinstance(v, SimpleNamespace) and v.optimize}

    # Generate ranges for each parameter that needs to be optimized
    param_ranges = {k: range(v.start, v.end + 1, v.step) for k, v in params_to_optimize.items()}

    # Generate all combinations of hyperparameters
    def generate_combinations(params):
        if not params:
            yield {}
            return
        param, values = params.popitem()
        for value in values:
            for combination in generate_combinations(params.copy()):
                combination[param] = value
                yield combination

    combinations = list(generate_combinations(param_ranges.copy()))

    # Create a new config for each combination
    config_list = []
    for combination in combinations:
        new_config = copy.deepcopy(config)
        for param, value in combination.items():
            setattr(new_config.hyper_parameters, param, SimpleNamespace(**{**vars(getattr(new_config.hyper_parameters, param)), "value": value}))
        config_list.append(new_config)

    return config_list



if __name__ == '__main__':    
  cfg_obj = check_and_get_configuration('./config/config.json', './config/config_schema.json')
  
  configs = generate_configs(cfg_obj)
  
  # Print or use the generated configurations
  for i, conf in enumerate(configs):
      print(f"Config {i + 1}:\n", conf)
