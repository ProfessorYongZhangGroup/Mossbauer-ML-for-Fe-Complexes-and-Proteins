#!/bin/bash
#SBATCH --job-name=ml-gaussian
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute-hf
#SBATCH --cpus-per-task=32
#SBATCH --mem=248GB
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=iskoulid@stevens.edu
#SBATCH --output=xtb.out

module purge

source /home/iskoulid/.bashrc
conda activate ml

source /home/iskoulid/.local/miniconda3/envs/ml/share/xtb/config_env.bash

export OMP_STACKSIZE=4G
export OMP_NUM_THREADS=8,1
export OMP_MAX_ACTIVE_LEVELS=1

export XTBPATH=/home/iskoulid/

cd new_optimized

for dir in ./*/
do
	cd $dir	
	#xtb --opt --gfnff --input param.inp *.mol
	#mv xtbopt.mol gfnff.mol
	xtb *.mol --alpb chloroform --opt --input param.inp *.mol > xtb.out
	tblite run --alpb chloroform --spin-polarized --post-processing xtbml_xyz xtbopt.mol
	cd ..
done

cd ..
