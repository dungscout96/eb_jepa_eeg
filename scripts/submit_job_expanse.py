from neurolab.jobs import Job

if __name__ == "__main__":
    # Define the job array with different configurations
    job = Job(
        name="eeg_jepa_experiments",          # job name + log file prefix
        cluster="expanse",                  # which cluster profile to use
        repo_path="/expanse/projects/nemar/dtyoung/eb_jepa_eeg",  # remote repo
        command="PYTHONPATH=. uv run --group eeg experiments/eeg_jepa/main.py", # entry point
        branch="",                        # which branch to use
        venv="__none__",                        
        env_vars={"WANDB_MODE": "online"},  # environment variables to set
    )
    script = job.submit(dry_run=True)
    print(script)
    job_id = job.submit()
    print(f"Submitted job with ID: {job_id}")
    