import json
from matplotlib import pyplot as plt
import argparse
import os
import pickle
import random
import json
import time

def generate_graph(models_outputs, hyper_param, graphs_dir, dummy=False, dots_format='.', score_prop='mean_episode_rewards'):
    timestamp = str(time.time()).split('.')[0]
    fig_path = os.path.join(graphs_dir, f'{hyper_param}_{timestamp}.jpeg')
    fig = plt.figure(figsize=(8,4), dpi=300)
    fig.suptitle('Training')
    plt.xlabel('timestep')
    plt.ylabel(f'{score_prop}')
    max_timestep = 4 * 10**6

    models_output = {}
    print(f"Generating {fig_path}")
    for model, output_path in models_outputs.items():
        statistics_output_path = os.path.join(output_path, 'stats')
        print(model)
        models_output[model] = {}
        if dummy:
            b = random.random() * random.random()
            a = 10 * random.random() * random.random()
            y = {'mean_episode_rewards' : [a + b * random.random() for i in range(10**5)]}
        else:
            TRY = 5
            for t in range(TRY):
                prop_path = os.path.join(statistics_output_path, f'{score_prop}.pkl')
                tmp_path = os.path.join(statistics_output_path, f'{score_prop}.tmp.pkl')
                os.system(f"cp -v {prop_path} {tmp_path}") 
                with open(tmp_path, 'rb') as f:
                    try:
                        print(f'Unpickle {tmp_path} try = {t}')
                        unpickler = pickle.Unpickler(f)
                        y = unpickler.load()
                        if score_prop == 'mean_loss':
                            try:
                                y.detach()
                            except:
                                pass
                        print(f'Unpickle {tmp_path} succeed')
                        break
                    except Exception as e:
                        print(f"Error whiile loading {tmp_path} (try={t})")
                        print(e)
            if t == TRY - 1:
                print(f'Skip {model}')
                continue

        with open(os.path.join(output_path, 'args.json'), 'rb') as f:
            models_output[model]['args'] = json.load(f)
            print(os.path.join(output_path, 'args.json'))

        y = y[:max_timestep]
        if score_prop == 'mean_loss':
            y = [0 if s == float('inf') else s.item() for s in y]
        plt.plot(list(range(len(y))), y, dots_format, label=f'{hyper_param}: {models_output[model]["args"][hyper_param]}')
        del y

    plt.legend(numpoints=1)
    plt.savefig(fig_path)
    print(f'Save fig: {fig_path}')
    #plt.show() 

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--graphs_dir', type=str, default='./graphs_output')
    parser.add_argument('--models', type=str)
    parser.add_argument('--prop', type=str)
    parser.add_argument('--dummy', type=bool, default=False)
    parser.add_argument('--dots_format', type=str, default='.')
    parser.add_argument('--score_prop', type=str, default='mean_episode_rewards')
    args = parser.parse_args()
    os.system(f'mkdir -p {args.graphs_dir}')

    with open(args.models, 'r') as f:
        models = f.read()
    models = models.split('\n')
    models = [m.strip() for m in models if m]
    models_output_paths = {model : os.path.join(args.results_dir, model) for model in models}
    generate_graph(models_output_paths, args.prop, args.graphs_dir, dummy=args.dummy, 
                   dots_format=args.dots_format, score_prop=args.score_prop)
    plt.figure().clear()

if __name__ == '__main__':
    main()
