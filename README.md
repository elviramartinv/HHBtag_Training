# HHBtag_Training

## Load environment on EL9 lxplus
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a_cuda/x86_64-el9-gcc11-opt/setup.sh

## Run Training.py
PAR=1; python3 -u Training.py -f=/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/*.root -params ../config/params_optimized_training.json -training_variables=../config/training_variables.json -patience=10 -validation_split=0.25 -n_epoch 1000 -parity $PAR --output model

## Run Apply.py
python3 Apply.py --file /afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/*.root --params_json ../config/params_optimized_training.json --training_variables ../config/training_variables.json --weights model1/model --parity 0