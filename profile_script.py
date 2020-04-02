from qmix import train

if __name__ == '__main__':
    import yaml
    from utilities import get_args
    args = get_args(yaml.load(open('default_config.yaml', 'r')))
    args.n_steps = 100
    train(args)