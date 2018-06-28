import os
import sys
import subprocess
import src.ts_config as config

def main():
    workdir = config.load_config('workdir')
    print(os.getcwd())
    sys.path.append(workdir)

    print('Gathering data')
    subprocess.call(['python3', 'src/query_transform.py'])
    print('Beginning Transformation and Fit')
    subprocess.call(['python3', 'src/fit.py'])

    print('Predicting . . .')
    subprocess.call(['python3', 'src/predict.py'])        


if __name__ == "__main__":
    main()