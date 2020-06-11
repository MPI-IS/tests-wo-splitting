#!/bin/bash
# navigate to the script directory to mae relative paths work
cd "$(dirname "$0")"

# Adapt the virtualenv to your needs. Make sure to install all requirements
source ~/PycharmProjects/tests-wo-splitting/venv/bin/activate
export PYTHONPATH="../"
for i in {0..34}
do
  python experiment.py --exp_number=$i
done



