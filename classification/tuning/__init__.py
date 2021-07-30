from ray import tune

def load_configs(exp_name):
    # Ray Option
    if exp_name == 'tune_1':
        config = {
                    'train/lr': tune.loguniform(1e-4, 1e-1)
                 }
    else:
        raise('Select Proper Exp-Name')

    return config
