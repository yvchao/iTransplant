import os
import pandas as pd
from sklearn.utils import check_random_state
from src.utils import train_evaluate, create_dir_if_not_exist

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def benchmark_one_center(model,
                         data_dir,
                         center,
                         scoring,
                         max_iter,
                         random_state,
                         hyper_parameter,
                         validation_split,
                         n_tests=10):
    random_state = check_random_state(random_state)
    seed_list = random_state.randint(1e8, size=(n_tests, ))
    scores = []
    for seed in seed_list:
        score = train_evaluate(model,
                               data_dir, [center],
                               seed,
                               scoring,
                               verbose=False,
                               discretize=False,
                               max_iter=max_iter,
                               validation_split=validation_split,
                               hyper_parameter=hyper_parameter)
        scores.append(score)
    stat = pd.DataFrame(scores)
    stat.loc[:, 'center'] = center
    return stat


def benchmark_multiple_centers(model,
                               data_dir,
                               centers,
                               scoring,
                               max_iter,
                               random_state,
                               hyper_parameter,
                               validation_split=0.2,
                               n_tests=10):
    stats = []
    for center in centers:
        stat = benchmark_one_center(model, data_dir, center, scoring, max_iter,
                                    random_state, hyper_parameter,
                                    validation_split, n_tests)
        stats.append(stat)
    return pd.concat(stats)


def benchmark(data_dir,
              report_dir,
              centers,
              configurations,
              scoring,
              n_tests=5,
              max_iter=200,
              seed=19260817):
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    create_dir_if_not_exist(report_dir)

    benchmark_results_file = os.path.join(report_dir, 'benchmarks.csv')

    tbar = tqdm.tqdm(configurations, position=0, leave=True)
    for i, configuration in enumerate(tbar):
        model = configuration['model']
        hyper_parameters = configuration['hyper_parameters']
        validation_split = configuration['validation_split']
        description = configuration['description']

        tbar.set_description(f'benchmark with {description} ...')
        records = []
        if os.path.exists(benchmark_results_file):
            existing_benchmarks = pd.read_csv(benchmark_results_file,
                                              index_col=0)
            if description in existing_benchmarks['description'].unique():
                continue
            else:
                records.append(existing_benchmarks)
        if description in ['LOWESS']:
            record = benchmark_multiple_centers(model,
                                                data_dir,
                                                centers,
                                                scoring,
                                                max_iter,
                                                seed,
                                                hyper_parameters,
                                                validation_split,
                                                n_tests=1)
        else:
            record = benchmark_multiple_centers(model, data_dir, centers,
                                                scoring, max_iter, seed,
                                                hyper_parameters,
                                                validation_split, n_tests)
        record.loc[:, 'description'] = description
        records.append(record)

        benchmarks = pd.concat(records, ignore_index=True)
        benchmarks.to_csv(benchmark_results_file)

    tbar.close()


def summarize_benchmark(report_dir, return_tables=True, use_redefined_row_order=True):
    benchmark_results_file = os.path.join(report_dir, 'benchmarks.csv')
    benchmark_results = pd.read_csv(benchmark_results_file, index_col=0)

    stats = []
    # Average performance on all centers
    for name, group in benchmark_results.groupby('description'):
        stat = group.describe()
        stat_dict = {}
        stat_dict['model'] = name
        for c in stat.columns:
            stat_dict[c] = f"{stat.loc['mean',c]:.3f}±{stat.loc['std',c]:.3f}"
        stats.append(stat_dict)

    stats = pd.DataFrame(stats)
    row_order = [
        'Logistic Regression', 'Per-cluster Logistic Regression',
        'Decision Tree', 'Per-cluster Decision Tree', 'LOWESS', 'LASSO',
        'Random Forest', 'INVASE', 'BC', 'iTransplant'
    ]
    if use_redefined_row_order:
        stats = stats.set_index('model').reindex(row_order)
    else:
        stats = stats.set_index('model')
    print('==== Summary of benchmark results ====')
    print(stats.to_markdown())
    if return_tables:
        return benchmark_results, stats
    else:
        return


def benchmark_visualization(benchmark_results, metric='AUC-PRC'):
    assert metric in ['AUC-ROC', 'AUC-PRC']
    df = pd.pivot_table(benchmark_results,
                        values=f'{metric}',
                        index='center',
                        columns=['description'],
                        aggfunc={f'{metric}': np.mean})
    fig, ax = plt.subplots(figsize=(16, 5))
    df.plot.bar(rot=0, ax=ax)
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    ax.set_xlabel('Centers')
    ax.set_ylabel(f'{metric}')
    fig.tight_layout()
    return fig


def center_wise_summary(selected_records, center, metric):
    df = pd.pivot_table(selected_records,
                        values=f'{metric}',
                        index='center',
                        columns=['description'],
                        aggfunc={f'{metric}': np.mean})
    means = df.loc[center]

    df = pd.pivot_table(selected_records,
                        values=f'{metric}',
                        index='center',
                        columns=['description'],
                        aggfunc={f'{metric}': np.std})
    stds = df.loc[center]

    df = pd.concat([means, stds], axis=1)
    df.columns = ['mean', 'std']
    df[metric] = df.apply(lambda row: f"{row['mean']:.3f}±{row['std']:.3f}",
                          axis=1)
    df = df[metric]
    return df
