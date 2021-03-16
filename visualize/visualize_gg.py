import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import click
import re
import sys

def get_sparsity(s):
    spars = re.search('sparsity=[0-9]+',s.strip())
    if spars is not None:
        spars = spars.group()
        return int(spars.split('=')[1])
    else:
        return None

def get_seed(s):
    spars = re.search('seed=[0-9]+',s.strip()).group()
    return int(spars.split('=')[1])

def get_task(s):
    spars = re.search('task=[0-9]+',s.strip()).group()
    return int(spars.split('=')[1])

def get_result(results):
    df = pd.DataFrame([]) # rows = sparisty, col = task
    previous_seed = None
    for i in range(1,len(results)):
        splits = results[i].split(',')
        name = splits[1].strip()
        sparsity = get_sparsity(name)
        task = get_task(name)
        
        seed = get_seed(name)
        if previous_seed is None:
            previous_seed = seed
        else:
            assert previous_seed == seed

        if sparsity is None:
            sparsity = 0
            assert task not in df.columns

        df.at[sparsity,task] = float(splits[-2]) # best val
            
    return df

def plot_figure(regular_df, weighted_df, supervised_df):
    columns = regular_df.columns.values
    assert np.array_equal(columns, weighted_df.columns.values)
    assert np.array_equal(columns, supervised_df.columns.values)

    index = regular_df.index.values
    assert np.array_equal(index, weighted_df.index.values)
    assert supervised_df.shape[0] == 1

    row_length=3
    fig, axarr = plt.subplots(regular_df.shape[0] // row_length + 
        int(bool(regular_df.shape[0] % row_length)), row_length, 
        figsize=(20,20))

    for i in range(len(axarr)):
        for j in range(len(axarr[i])):
            sparsity = regular_df.index[j + (i * row_length)]
            axarr[i,j].plot(columns, regular_df.loc[sparsity], 
                    label = 'regular')
            axarr[i,j].plot(columns, weighted_df.loc[sparsity], 
                    label = 'weighted')
            axarr[i,j].plot(columns, supervised_df.loc[0], 
                    label = 'upper bound')
            axarr[i,j].set_xlabel('Task ID')
            axarr[i,j].set_ylabel('Val Acc')
            axarr[i,j].set_title('Sparsity - {}'.format(sparsity))
            axarr[i,j].legend(loc='best')
            axarr[i,j].set_ylim(0.65,1.0)
    fig.suptitle('GG Experiment on SplitCifar100')
    plt.show()

@click.command()
@click.option(
    '--regular_results',
    '-r',
    'regular_results',
    default=None,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False
    )   
)
@click.option(
    '--weighted_results',
    '-w',
    'weighted_results',
    default=None,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False
    )   
)
@click.option(
    '--supervised_results',
    '-s',
    'supervised_results',
    default=None,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False
    )   
)
def main(regular_results, weighted_results, supervised_results):
    with open(regular_results) as f:
        regular = f.readlines()

    with open(weighted_results) as f:
        weighted = f.readlines()

    with open(supervised_results) as f:
        supervised = f.readlines()

    regular_df = get_result(regular)
    weighted_df = get_result(weighted)
    supervised_df = get_result(supervised)

    plot_figure(regular_df, weighted_df,supervised_df)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
