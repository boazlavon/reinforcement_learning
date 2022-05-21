import json
from matplotlib import pyplot as plt
import argparse
import os
import pickle
import random
import json
import time

def generate_graph(models_outputs, graphs_dir, score_props, dots_format='.'):
    timestamp = str(time.time()).split('.')[0]
    fig_path = os.path.join(graphs_dir, f'q1_{timestamp}.jpeg')
    fig = plt.figure(figsize=(8,4), dpi=300)
    fig.suptitle('Q1 Training Plot')
    plt.xlabel('timestep')
    plt.ylabel(f'task reward')
    max_timestep = 6 * 10**6

    models_output = {}
    print(f"Generating {fig_path}")
    for model, output_path in models_outputs.items():
        statistics_output_path = os.path.join(output_path, 'stats')
        print(model)
        models_output[model] = {}
        with open(os.path.join(output_path, 'args.json'), 'rb') as f:
            models_output[model]['args'] = json.load(f)
            print(os.path.join(output_path, 'args.json'))

        for score_prop in score_props:
            score_prop_path = os.path.join(statistics_output_path, f'{score_prop}.pkl')
            tmp_path = os.path.join(statistics_output_path, f'{score_prop}.tmp.pkl')
            os.system(f"cp -v {score_prop_path} {tmp_path}") 
            with open(tmp_path, 'rb') as f:
                try:
                    print(f'Unpickle {tmp_path}')
                    unpickler = pickle.Unpickler(f)
                    y = unpickler.load()
                    if score_prop == 'mean_loss':
                        try:
                            y.detach()
                        except:
                            pass
                    print(f'Unpickle {tmp_path} succeed')
                except Exception as e:
                    print(f"Error whiile loading {tmp_path}")
                    print(e)

            y = y[:max_timestep]
            if score_prop == 'mean_loss':
                y = [0 if s == float('inf') else s.item() for s in y]

            plt.plot(list(range(len(y))), y, dots_format, label=f'{score_prop}')
            del y

    plt.legend(numpoints=1)
    plt.savefig(fig_path)
    print(f'Save fig: {fig_path}')

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--graphs_dir', type=str, default='./graphs_output')
    parser.add_argument('--models', type=str)
    args = parser.parse_args()
    os.system(f'mkdir -p {args.graphs_dir}')

    with open(args.models, 'r') as f:
        models = f.read()
    models = models.split('\n')
    model = [m.strip() for m in models if m][0]
    models_output_paths = {model : os.path.join(args.results_dir, model) }
    #score_props = ['best_mean_episode_rewards', 'mean_episode_rewards']
    score_props = ['mean_loss']
    generate_graph(models_output_paths, args.graphs_dir, score_props)
    plt.figure().clear()

if __name__ == '__main__':
    main()
