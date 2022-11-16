


import subprocess
import argparse



def download(dataset):
    cmd = ['python', 'get_dataset_script/get_{}.py'.format(dataset)]
    print(' '.join(cmd))
    subprocess.call(cmd)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets for MIML-DA.')
    parser.add_argument('--dataset', metavar='N', type=str, nargs='+', 
                    choices=['aircraft', 'bird', 'cifar', 'miniimagenet'])
    parser.parse_args(['--dataset', 'aircraft', 'bird', 'cifar', 'miniimagenet'])
    args = ['aircraft', 'bird', 'cifar', 'miniimagenet']
    args = ['miniimagenet']
    if len(args) > 0:
        for dataset in args:
            download(dataset)
