target=AMD

#workdir=/mnt/ubudata/projects/forecast-engine
workdir=/mnt/raid0/Projects/forecast-engine
n_predictors=50
n_in=361
n_out=361

cd $workdir
# prep data
python daily_stock/query_transform.py $target $n_predictors

# run predict for next day
cd src/
python predict.py $target $n_in $n_out $n_predictors
