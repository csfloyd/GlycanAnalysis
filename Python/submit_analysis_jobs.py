import os
import shutil

n_params = 1
dir = "ff_scan_pr0"
inputs_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/" + dir + "/"
results_base = "/project/svaikunt/csfloyd/TrainingCRNs/AnalyzedData/" + dir + "/"

# Define the range of values for param1 and labels for param2

param1_values = ["1", "2", "3", "4", "11", "22", "33", "44"]
param2_values = [1,2,3]

# SLURM job template
job_template = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/CRN_training.out   # Redirect stdout to the output directory
#SBATCH --error={output}/CRN_training.err    # Redirect stderr to the output directory
#SBATCH --time=32:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 /project/svaikunt/csfloyd/TrainingCRNs/Python/run_analysis.py --param1 {param1} --input {input_dir} --output {output}
"""

if n_params == 1:
    # Create results directory if it doesn't exist
    if not os.path.exists(results_base):
        os.makedirs(results_base)
        print(f"Created results directory: {results_base}")
    
    # Loop over different parameter values
    for param1 in param1_values:
        input_dir = inputs_base + f"{param1}"  # Input data directory (existing)
        output = results_base + f"{param1}"    # Results output directory
        
        # Create results output directory if it doesn't exist
        if not os.path.exists(output):
            os.makedirs(output)
            print(f"Created results directory: {output}")
        else:
            print(f"Results directory already exists: {output}")

        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory does not exist: {input_dir}")
            continue

        # Generate job script content
        job_script_content = job_template.format(param1=param1, input_dir=input_dir, output=output)

        # Define a unique job filename in the results directory
        job_filename = os.path.join(output, f"job_{param1}.sh")

        # Write the job script to a file
        with open(job_filename, "w") as job_file:
            job_file.write(job_script_content)

        # Submit the job using sbatch
        os.system(f"sbatch {job_filename}")

        print(f"Submitted analysis job with param1={param1}, input={input_dir}, output={output}")



# SLURM job template
job_template_2 = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/CRN_training.out   # Redirect stdout to the output directory
#SBATCH --error={output}/CRN_training.err    # Redirect stderr to the output directory
#SBATCH --time=32:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 /project/svaikunt/csfloyd/TrainingCRNs/Python/run_analysis.py --param1 {param1} --param2 {param2} --input {input_dir} --output {output}
"""

if n_params == 2:
    # Create results directory if it doesn't exist
    if not os.path.exists(results_base):
        os.makedirs(results_base)
        print(f"Created results directory: {results_base}")
    
    # Loop over different parameter values
    for param1 in param1_values:
        for param2 in param2_values:
            input_dir = os.path.join(inputs_base, f"{param1}_{param2}")  # Input data directory (existing)
            output = os.path.join(results_base, f"{param1}_{param2}")    # Results output directory
            
            # Create results output directory if it doesn't exist
            if not os.path.exists(output):
                os.makedirs(output)
                print(f"Created results directory: {output}")
            else:
                print(f"Results directory already exists: {output}")

            # Check if input directory exists
            if not os.path.exists(input_dir):
                print(f"Warning: Input directory does not exist: {input_dir}")
                continue

            # Generate job script content
            job_script_content = job_template_2.format(param1=param1, param2=param2, input_dir=input_dir, output=output)

            # Define a unique job filename inside the results directory
            job_filename = os.path.join(output, f"job_{param1}_{param2}.sh")

            # Write the job script to a file
            with open(job_filename, "w") as job_file:
                job_file.write(job_script_content)

            # Submit the job using sbatch
            os.system(f"sbatch {job_filename}")

            print(f"Submitted analysis job with param1={param1}, param2={param2}, input={input_dir}, output={output}")

