import os
import shutil
import numpy as np

# Training script to run
training_script = "/project/svaikunt/csfloyd/TrainingCRNs/Python/run_training.py"

n_params = 3
output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/NR_width_seed_big_pr1/"
#output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/width_pf_seed_big_pr0_2/"
#output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/noise_scale_clb_seed_mi_output2/"
output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/noise_scale_hidden_dim_seed_mi_outputM/"
output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/log_variance_hidden_dim_seed_mi_outputM/"
output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/pf_noise_scale_seed_mi_output10/"
output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/log_variance_pf_seed2/"
#output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/tasks_pf_seed_big/"

# Define the range of values for param1 and labels for param2

param1_values = [5, 10, 15, 20]
#param2_values = [1,2,3,4,5]
param2_values = [0.25, 0.5, 0.75, 1.0]
param1_values = [0.0, 10.0, 20.0]
param1_values = [2, 3, 4]
param1_values = 0.25 * np.array([1e-10, 0.25, 0.5, 0.75, 1])
param1_values = [0.25, 0.5, 0.75, 1.0]
param1_values = [0.5, 1.0, 1.5, 2.0]
param1_values = [0.1, 0.2, 0.3, 0.4, 0.5]

param2_values = [1e-8, 0.1, 0.2, 0.3, 0.4, 0.5]
param2_values = [1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
#param2_values = [2, 4, 6, 8]

param1_values = [0.0, 0.05, 0.1, 0.15, 0.2]
param2_values = [0.2, 0.4, 0.6, 0.8, 1.0]
param3_values = [1, 2, 3]




# SLURM job template
job_template = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/CRN_training.out   # Redirect stdout to the output directory
#SBATCH --error={output}/CRN_training.err    # Redirect stderr to the output directory
#SBATCH --time=2:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16000

# module load python3

python3 {training_script} --param1 {param1} --output {output}
"""

if n_params == 1:
    # Loop over different parameter values
    for param1 in param1_values:
        output = output_base + f"{param1}"  # Define output folder name

        # Remove existing directory if it exists, then recreate it
        if os.path.exists(output):
            shutil.rmtree(output)  # Delete existing directory and contents
        os.makedirs(output)  # Create a new empty directory

        print(f"Created directory: {output}")

        # Generate job script content
        job_script_content = job_template.format(training_script=training_script, param1=param1, output=output)

        # Define a unique job filename
        job_filename = os.path.join(output, f"job_{param1}.sh")

        # Write the job script to a file
        with open(job_filename, "w") as job_file:
            job_file.write(job_script_content)

        # Submit the job using sbatch
        os.system(f"sbatch {job_filename}")

        print(f"Submitted job with param1={param1} and output={output}")



# SLURM job template
job_template_2 = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/CRN_training.out   # Redirect stdout to the output directory
#SBATCH --error={output}/CRN_training.err    # Redirect stderr to the output directory
#SBATCH --time=6:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 {training_script} --param1 {param1} --param2 {param2} --output {output}
"""

if n_params == 2:
    # Loop over different parameter values
    for param1 in param1_values:
        for param2 in param2_values:
            output = os.path.join(output_base, f"{param1}_{param2}")  # Unique output folder for each param1, param2 combination

            # Remove existing directory if it exists, then recreate it
            if os.path.exists(output):
                shutil.rmtree(output)  # Delete existing directory and contents
            os.makedirs(output)  # Create a new empty directory

            print(f"Created directory: {output}")

            # Generate job script content
            job_script_content = job_template_2.format(training_script=training_script, param1=param1, param2=param2, output=output)

            # Define a unique job filename inside the output directory
            job_filename = os.path.join(output, f"job_{param1}_{param2}.sh")

            # Write the job script to a file
            with open(job_filename, "w") as job_file:
                job_file.write(job_script_content)

            # Submit the job using sbatch
            os.system(f"sbatch {job_filename}")

            print(f"Submitted job with param1={param1}, param2={param2}, and output={output}")


# SLURM job template for 3 params
job_template_3 = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/CRN_training.out   # Redirect stdout to the output directory
#SBATCH --error={output}/CRN_training.err    # Redirect stderr to the output directory
#SBATCH --time=0:30:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 {training_script} --param1 {param1} --param2 {param2} --param3 {param3} --output {output}
"""

if n_params == 3:
    # Loop over different parameter values
    for param1 in param1_values:
        for param2 in param2_values:
            for param3 in param3_values:
                output = os.path.join(output_base, f"{param1}_{param2}_{param3}")  # Unique output folder for each param combination

                # Remove existing directory if it exists, then recreate it
                if os.path.exists(output):
                    shutil.rmtree(output)  # Delete existing directory and contents
                os.makedirs(output)  # Create a new empty directory

                print(f"Created directory: {output}")

                # Generate job script content
                job_script_content = job_template_3.format(training_script=training_script, param1=param1, param2=param2, param3=param3, output=output)

                # Define a unique job filename inside the output directory
                job_filename = os.path.join(output, f"job_{param1}_{param2}_{param3}.sh")

                # Write the job script to a file
                with open(job_filename, "w") as job_file:
                    job_file.write(job_script_content)

                # Submit the job using sbatch
                os.system(f"sbatch {job_filename}")

                print(f"Submitted job with param1={param1}, param2={param2}, param3={param3}, and output={output}")

