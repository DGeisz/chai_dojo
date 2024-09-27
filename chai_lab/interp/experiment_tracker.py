import wandb

from git import Repo
from typing import List


class ExperimentTracker:
    run = None

    def __init__(self):
        self.repo = Repo(search_parent_directories=True)
        wandb.login()

    def new_experiment(
        self,
        project: str,
        experiment_description: str,
        config: dict,
        only_commit_tracked_files=False,
        **kwargs,
    ):
        if self.run is not None:
            self.run.finish()

        files_to_commit: List[str] = [
            item.a_path for item in self.repo.index.diff(None)
        ]

        if not only_commit_tracked_files:
            files_to_commit.extend(self.repo.untracked_files)

        print("Files to commit:", files_to_commit)

        if any(not f.endswith(".py") for f in files_to_commit):
            raise ValueError(
                "There are changes to non-python files! Handle these before you run a new experiment."
            )

        # Add all files
        self.repo.index.add(files_to_commit)

        self.repo.index.commit(f"EXP: {experiment_description}")

        # Extend config with the commit hash
        config["commit"] = self.repo.head.commit.hexsha

        self.run = wandb.init(
            project=project, config=config, notes=experiment_description, **kwargs
        )

        return self.run

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
        else:
            print("No run to finish!")
