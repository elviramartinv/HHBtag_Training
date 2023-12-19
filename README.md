# HHBtag_Training

## Load environment on EL9 lxplus
```sh
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.30.00/x86_64-centos9-gcc113-opt/bin/thisroot.sh
export MAMBA_ROOT_PREFIX=/afs/cern.ch/work/k/kandroso/micromamba
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"
micromamba activate hh
```

## Run Training.py
PAR=1; python3 -u Training.py -f=/afs/cern.ch/user/e/emartinv/public/Ntuples_prod/training_ntuples/*.root -params ../config/params_optimized_training.json -training_variables=../config/training_variables.json -patience=10 -validation_split=0.25 -n_epoch 1000 -parity $PAR --output model

## Run Apply.py
python3 Apply.py --file ../input/GluGluToHHTo2B2Tau_node_SM.root --params_json ../config/params_optimized_training.json --training_variables ../config/training_variables.json --weights ../models/HHBtag_13nov_par1/model --parity 0

## Training evaluation
### Run PermutationFeatureImportance.py
python3 PermutationFeatureImportance.py --parity 1 --weights model/model --training_variables ../config/training_variables.json --params_json ../config/params_optimized_training.json --file /afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/*.root
