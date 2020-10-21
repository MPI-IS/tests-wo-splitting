# This target installs the package through pip
install:
	pip install .

# For the experiment results we are going to use a tricky approach:
# for each data file we are going to generate a separate target in
# runtime, and then a PHONY target `data` that encapsulates all of
# them. This approach allows us to re-use the data files and to
# rebuild only the ones that are missing.

# All the data files should be searched in a separate directory.
vpath %.data data/

# The experiment is defined by its number. Here we generate all the
# experiment numbers and store them in a variable.
EXP_NUMS = $(shell seq 0 29)

# This macro generates a target named results_N.data.
define make-experiment-result-target
$(addsuffix .data, $(addprefix results_, $1)): config.yml
	python scripts/experiment.py --exp_number=$(strip $1)
endef

# We call the macro for each number in $EXP_NUMS
$(foreach num, $(EXP_NUMS), $(eval $(call make-experiment-result-target, $(num))))

# And put all the targets into one common target `data`
.PHONY: data
data: $(addsuffix .data, $(addprefix results_, $(EXP_NUMS)))


# This target rebuilds Figure 2 from the paper and puts into the root
# the project.
evaluation.pdf: data
	python scripts/evaluation.py

# Short alias
.PHONY: fig
fig: evaluation.pdf
