"""Convergence study for the Bertschinger shell-ensemble caustic estimate."""

from copy import deepcopy
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bertschinger_shell_ode_comparison import run_comparison
import tools as example_tools


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'bertschinger_reference.yaml'
OUTPUT = ROOT / 'outputs_caustic_convergence'


CASES = {
    'shells': [2048, 4096, 8192],
    'smoothing': [6.0, 12.0, 24.0],
    'radius_inner_dimensionless': [0.025, 0.05, 0.1],
    'perturbation_amplitude': [4.0, 5.0, 6.0],
}


def _config(par_config, initial_condition, example, filename):
    with filename.open('w') as handle:
        yaml.safe_dump({'par': {'simulation': {'name': par_config['simulation']['name']},
                                'output': {'directory': '.', 'savedir': par_config['output']['savedir']},
                                'units': {'CodeUnits': par_config['units']['CodeUnits']}},
                        'initial_condition': initial_condition, 'example': example}, handle,
                       sort_keys=False)


def main():
    base_config = example_tools.load_reference_config(CONFIG)
    base_par_config = base_config['par']
    base_initial_condition = base_config['initial_condition']
    base_example = base_config['example']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='radhydropy-caustic-') as temp:
        temp = Path(temp)
        for parameter, values in CASES.items():
            for value in values:
                par_config = deepcopy(base_par_config)
                initial_condition = deepcopy(base_initial_condition)
                example = deepcopy(base_example)
                if parameter == 'shells':
                    initial_condition['number_of_shells'] = value
                elif parameter == 'smoothing':
                    example['caustic_smoothing_bins'] = value
                else:
                    initial_condition[parameter] = value
                label = '%s_%s' % (parameter, str(value).replace('.', 'p'))
                par_config['output']['savedir'] = str(OUTPUT / label)
                Path(par_config['output']['savedir']).mkdir(parents=True, exist_ok=True)
                config = temp / (label + '.yaml')
                _config(par_config, initial_condition, example, config)
                run_comparison(config)
                data = np.load(Path(par_config['output']['savedir']) /
                                'BertschingerDarkMatterCaustic.npz')
                selected = data['lambda_caustic'][data['xi'] >= 3.0]
                rows.append((parameter, float(value), selected.size,
                             np.median(selected), np.std(selected),
                             float(data['ode_outer_caustic_lambda'])))

    dtype = [('parameter', 'U32'), ('value', 'f8'), ('samples', 'i4'),
             ('median_lambda', 'f8'), ('scatter_lambda', 'f8'),
             ('ode_lambda', 'f8')]
    result = np.array(rows, dtype=dtype)
    np.save(OUTPUT / 'caustic_convergence.npy', result)
    with (OUTPUT / 'caustic_convergence.txt').open('w') as handle:
        handle.write('parameter value samples median_lambda scatter_lambda '
                     'ode_lambda offset\n')
        for row in result:
            handle.write('%s %.8g %d %.8g %.8g %.8g %.8g\n' %
                         (*row, row['median_lambda'] - row['ode_lambda']))

    figure, axes = plt.subplots(1, len(CASES), figsize=(15, 3.5), sharey=True)
    for axis, parameter in zip(axes, CASES):
        selected = result[result['parameter'] == parameter]
        axis.errorbar(selected['value'], selected['median_lambda'],
                      yerr=selected['scatter_lambda'], marker='o',
                      capsize=3, color='tab:red')
        axis.axhline(selected['ode_lambda'][0], color='black', linestyle='--')
        axis.set_title(parameter)
        axis.set_xlabel('value')
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r'late-time $\lambda_{\rm caustic}$')
    figure.tight_layout()
    figure.savefig(OUTPUT / 'BertschingerCausticConvergence.jpg', dpi=180)
    plt.close(figure)
    print('wrote %s' % (OUTPUT / 'caustic_convergence.txt'))


if __name__ == '__main__':
    main()
