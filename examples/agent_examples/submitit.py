import submitit
import subprocess


def run():
    subprocess.run(
        ["python", "whittington_2020_run.py"],
        check=True
    )


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="submitit_logs")

    executor.update_parameters(
        slurm_partition="bigbatch",
        cpus_per_task=28,
        timeout_min=720,
        slurm_job_name="simple_run",
    )

    job = executor.submit(run)

    print(f"Submitted job {job.job_id}")