import os
import numpy as np

# Worker script (cluster path)
analysis_script = "/project/svaikunt/csfloyd/TrainingCRNs/Python/run_training_analysis.py"

n_params = 3

# Must match the training sweep folder name under Dirs/Training/
sweep_name = "Tsoft_hidden_dim_seed3"
#sweep_name = "pf_noise_scale_seed_mi_output10"
#sweep_name = "log_variance_hidden_dim_seed_mi_outputM"

input_base = f"/project/svaikunt/csfloyd/TrainingCRNs/Dirs/Training/{sweep_name}/"
output_base = f"/project/svaikunt/csfloyd/TrainingCRNs/AnalyzedData/Training/{sweep_name}/"

# Match the training job parameter grids
param1_values = [0.0, 0.05, 0.1, 0.15, 0.2]
param2_values = [0.2, 0.4, 0.6, 0.8, 1.0]
param1_values = [0.1, 0.5, 1.0, 2.0, 5.0]
param2_values = [1,2,3,4]
param3_values = [1, 2, 3]

n_samples_per_mode = 100
plot_seed = 0


job_template_1 = """#!/bin/bash
#SBATCH --job-name=train_analysis
#SBATCH --output={output}/CRN_analysis.out
#SBATCH --error={output}/CRN_analysis.err
#SBATCH --time=1:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

python3 {analysis_script} --param1 {param1} --input {input_dir} --output {output} --n_samples_per_mode {n_samples_per_mode} --plot_seed {plot_seed}
"""

job_template_2 = """#!/bin/bash
#SBATCH --job-name=train_analysis
#SBATCH --output={output}/CRN_analysis.out
#SBATCH --error={output}/CRN_analysis.err
#SBATCH --time=1:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

python3 {analysis_script} --param1 {param1} --param2 {param2} --input {input_dir} --output {output} --n_samples_per_mode {n_samples_per_mode} --plot_seed {plot_seed}
"""

job_template_3 = """#!/bin/bash
#SBATCH --job-name=train_analysis
#SBATCH --output={output}/CRN_analysis.out
#SBATCH --error={output}/CRN_analysis.err
#SBATCH --time=1:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

python3 {analysis_script} --param1 {param1} --param2 {param2} --param3 {param3} --input {input_dir} --output {output} --n_samples_per_mode {n_samples_per_mode} --plot_seed {plot_seed}
"""


def _has_training_pickle(input_dir):
    return (
        os.path.isfile(os.path.join(input_dir, "mi_training_results.pkl"))
        or os.path.isfile(os.path.join(input_dir, "training_results.pkl"))
    )


def _prepare_and_submit(input_dir, output, job_filename, job_script_content):
    if not os.path.exists(input_dir):
        print(f"Warning: input directory does not exist: {input_dir}")
        return False
    if not _has_training_pickle(input_dir):
        print(f"Warning: no training pickle in: {input_dir}")
        return False

    os.makedirs(output, exist_ok=True)
    with open(job_filename, "w") as job_file:
        job_file.write(job_script_content)
    os.system(f"sbatch {job_filename}")
    return True


os.makedirs(output_base, exist_ok=True)
print(f"Input base:  {input_base}")
print(f"Output base: {output_base}")

if n_params == 1:
    for param1 in param1_values:
        input_dir = os.path.join(input_base, f"{param1}")
        output = os.path.join(output_base, f"{param1}")
        job_filename = os.path.join(output, f"job_{param1}.sh")
        job_script_content = job_template_1.format(
            analysis_script=analysis_script,
            param1=param1,
            input_dir=input_dir,
            output=output,
            n_samples_per_mode=n_samples_per_mode,
            plot_seed=plot_seed,
        )
        if _prepare_and_submit(input_dir, output, job_filename, job_script_content):
            print(f"Submitted analysis job param1={param1}")

elif n_params == 2:
    for param1 in param1_values:
        for param2 in param2_values:
            input_dir = os.path.join(input_base, f"{param1}_{param2}")
            output = os.path.join(output_base, f"{param1}_{param2}")
            job_filename = os.path.join(output, f"job_{param1}_{param2}.sh")
            job_script_content = job_template_2.format(
                analysis_script=analysis_script,
                param1=param1,
                param2=param2,
                input_dir=input_dir,
                output=output,
                n_samples_per_mode=n_samples_per_mode,
                plot_seed=plot_seed,
            )
            if _prepare_and_submit(input_dir, output, job_filename, job_script_content):
                print(f"Submitted analysis job param1={param1}, param2={param2}")

elif n_params == 3:
    for param1 in param1_values:
        for param2 in param2_values:
            for param3 in param3_values:
                input_dir = os.path.join(input_base, f"{param1}_{param2}_{param3}")
                output = os.path.join(output_base, f"{param1}_{param2}_{param3}")
                job_filename = os.path.join(output, f"job_{param1}_{param2}_{param3}.sh")
                job_script_content = job_template_3.format(
                    analysis_script=analysis_script,
                    param1=param1,
                    param2=param2,
                    param3=param3,
                    input_dir=input_dir,
                    output=output,
                    n_samples_per_mode=n_samples_per_mode,
                    plot_seed=plot_seed,
                )
                if _prepare_and_submit(input_dir, output, job_filename, job_script_content):
                    print(
                        f"Submitted analysis job "
                        f"param1={param1}, param2={param2}, param3={param3}"
                    )

else:
    raise ValueError(f"Unsupported n_params={n_params}")
