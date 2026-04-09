import os
import shutil
import numpy as np

n_params = 3
output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/NR_width_seed_big_pr1/"
#output_base = "/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/tasks_width_seed_big/"

# Define the range of values for param1 and labels for param2

param1_values = [1,2,3,4,5]
param2_values = [1,2,3,4,5]
param3_values = [1,2,3,4,5]



# SLURM job template
job_template = """#!/bin/bash
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

python3 /project/svaikunt/csfloyd/TrainingCRNs/Python/run_training.py --param1 {param1} --output {output}
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
        job_script_content = job_template.format(param1=param1, output=output)

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

python3 /project/svaikunt/csfloyd/TrainingCRNs/Python/run_training.py --param1 {param1} --param2 {param2} --output {output}
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
            job_script_content = job_template_2.format(param1=param1, param2=param2, output=output)

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
#SBATCH --time=6:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 /project/svaikunt/csfloyd/TrainingCRNs/Python/run_training.py --param1 {param1} --param2 {param2} --param3 {param3} --output {output}
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
                job_script_content = job_template_3.format(param1=param1, param2=param2, param3=param3, output=output)

                # Define a unique job filename inside the output directory
                job_filename = os.path.join(output, f"job_{param1}_{param2}_{param3}.sh")

                # Write the job script to a file
                with open(job_filename, "w") as job_file:
                    job_file.write(job_script_content)

                # Submit the job using sbatch
                os.system(f"sbatch {job_filename}")

                print(f"Submitted job with param1={param1}, param2={param2}, param3={param3}, and output={output}")

