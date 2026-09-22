import os
import shutil
import numpy as np

n_params = 3

dir = "n_paths_sample_idx_seed_pr0"
inputs_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Sampling/" + dir + "/"
results_base = "/project/svaikunt/csfloyd/TrainingCRNs/AnalyzedData/Sampling/" + dir + "/"

# Match submit_scanning_jobs.py: n_paths, sample idx, p_r
param1_values = [1,2,3,4,5] # n_paths
param2_values = range(10) # graph samples
param3_values = range(10) # seeds

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

python3 /project/svaikunt/csfloyd/TrainingCRNs/Python/run_sampling_analysis.py --param1 {param1} --input {input_dir} --output {output}
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

python3 /project/svaikunt/csfloyd/TrainingCRNs/Python/run_sampling_analysis.py --param1 {param1} --param2 {param2} --input {input_dir} --output {output}
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


# SLURM job template
job_template_3 = """#!/bin/bash
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

python3 /project/svaikunt/csfloyd/TrainingCRNs/Python/run_sampling_analysis.py --param1 {param1} --param2 {param2} --param3 {param3} --input {input_dir} --output {output}
"""

if n_params == 3:
    if not os.path.exists(results_base):
        os.makedirs(results_base)
        print(f"Created results directory: {results_base}")

    for param1 in param1_values:
        for param2 in param2_values:
            for param3 in param3_values:
                input_dir = os.path.join(inputs_base, f"{param1}_{param2}_{param3}")
                output = os.path.join(results_base, f"{param1}_{param2}_{param3}")

                if not os.path.exists(output):
                    os.makedirs(output)
                    print(f"Created results directory: {output}")
                else:
                    print(f"Results directory already exists: {output}")

                if not os.path.exists(input_dir):
                    print(f"Warning: Input directory does not exist: {input_dir}")
                    continue
                if not os.path.isfile(os.path.join(input_dir, "SavedData.pkl")):
                    print(f"Warning: no SavedData.pkl in: {input_dir}")
                    continue

                job_script_content = job_template_3.format(
                    param1=param1, param2=param2, param3=param3,
                    input_dir=input_dir, output=output
                )

                job_filename = os.path.join(output, f"job_{param1}_{param2}_{param3}.sh")

                with open(job_filename, "w") as job_file:
                    job_file.write(job_script_content)

                os.system(f"sbatch {job_filename}")

                print(
                    f"Submitted analysis job with param1={param1}, param2={param2}, "
                    f"param3={param3}, input={input_dir}, output={output}"
                )

