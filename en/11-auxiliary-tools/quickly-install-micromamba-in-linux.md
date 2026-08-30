# Quickly Install Micromamba In Linux

sudo apt install bzip2



curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
#./bin/micromamba shell init -s bash -p /opt/micromamba
The latest version no longer supports the -p command.

./bin/micromamba shell init -s bash --root-prefix ./micromamba

vi ~/.bashrc
Add
alias mamba=micromamba


Then configure .condarc.
