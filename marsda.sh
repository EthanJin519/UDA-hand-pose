#!/bin/bash
#BATCH -A test                  
#SBATCH -J mars              
# SBATCH -N 1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=10      
#SBATCH --gres=gpu:1           
#SBATCH -p p40                 
#SBATCH -t 2-12:00:00           
#SBATCH -o mars      

conda activate base            
# 运行代码
python marsda.py

