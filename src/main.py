import os
import sys
import subprocess
import config


def main():
    workdir = config.load_config('workdir')
    print(os.getcwd())
    sys.path.append(workdir)

    
    os.chdir(workdir)
    print('Gathering data')
    subprocess.call(['python3', 'daily_stock/query_transform.py'])
    print('Beginning Transformation and Fit')
    subprocess.call(['python3', 'src/fit.py'])

    os.chdir('./src')
    print('Predicting . . .')
    subprocess.call(['python3', 'predict.py'])        


if __name__ == "__main__":
    main()