from types import SimpleNamespace
import copy
from config_helper import check_and_get_configuration
from net_runner import NetRunner
from visual_util import ColoredPrint as cp

class AutoOptimizer:
    def __init__(self, cfg_object: object) -> None:
        self.configurations = self.__generate_configs(cfg_object)
        
    def run_optimizations(self):
        # Iterate over each configuration and perform training and testing
        for i, cfg_obj in enumerate(self.configurations):
            cp.blue(f"\nRunning optimization for configuration {i + 1} / {len(self.configurations)}")
            
            # Create the object that will allow training and testing the model
            runner = NetRunner(cfg_obj)
            
            # Perform training if requested
            if cfg_obj.parameters.train:
                runner.train()
            
            # Perform testing if requested
            if cfg_obj.parameters.test:
                runner.test(preview=True, print_loss=True)

    # Function to recursively convert dictionaries and lists into SimpleNamespace objects
    def __dict_to_simplenamespace(self, d):
        if isinstance(d, dict):
            # Convert each key-value pair in the dictionary
            for k, v in d.items():
                d[k] = self.__dict_to_simplenamespace(v)
            # Convert the dictionary itself into a SimpleNamespace
            return SimpleNamespace(**d)
        elif isinstance(d, list):
            # If the object is a list, convert each element in the list
            return [self.__dict_to_simplenamespace(i) for i in d]
        else:
            # If the object is neither a dictionary nor a list, return it as is
            return d

    # Function to generate all combinations of hyperparameter configurations
    def __generate_configs(self, cfg_obj):
        # Convert the input configuration object into SimpleNamespace format
        config = self.__dict_to_simplenamespace(cfg_obj)
        
        # Extract the hypernet_parameters from the configuration
        hyper_params = config.net_parameters
        
        # Identify the hypernet_parameters that need to be optimized
        params_to_optimize = {k: v for k, v in vars(hyper_params).items() if isinstance(v, SimpleNamespace) and v.optimize}

        # Generate ranges for each hyperparameter that needs to be optimized
        param_ranges = {k: range(v.start, v.end + 1, v.step) for k, v in params_to_optimize.items()}

        # Recursive function to generate all combinations of hyperparameter values
        def generate_combinations(params):
            if not params:
                # Base case: if no more net_parameters, yield an empty dictionary
                yield {}
                return
            # Pop an item from the parameter dictionary
            param, values = params.popitem()
            for value in values:
                # Recursively generate combinations for the remaining net_parameters
                for combination in generate_combinations(params.copy()):
                    combination[param] = value
                    yield combination

        # Generate all possible combinations of hypernet_parameters
        combinations = list(generate_combinations(param_ranges.copy()))

        # Create a new configuration object for each combination of hypernet_parameters
        config_list = []
        for combination in combinations:
            # Deep copy the original configuration to create a new one
            new_config = copy.deepcopy(config)
            for param, value in combination.items():
                # Update the hyperparameter values in the new configuration
                setattr(new_config.net_parameters, param, SimpleNamespace(**{**vars(getattr(new_config.net_parameters, param)), "value": value}))
            config_list.append(new_config)

        return config_list

# Main script execution
if __name__ == '__main__':    
    # Load the configuration object and schema from JSON files
    cfg_object = check_and_get_configuration('./config/config.json', './config/config_schema.json')
    
    # Initialize AutoOptimizer obj
    autoOptimizer = AutoOptimizer(cfg_object)
    
    # Print or use the generated configurations
    for i, conf in enumerate(autoOptimizer.configurations):
        print(f"Config {i + 1}:\n", conf)
