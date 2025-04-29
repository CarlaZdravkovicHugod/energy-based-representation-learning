"""
Neptune Logger for logging metrics, models, and metadata to Neptune.
"""

import neptune
import os

class NeptuneLogger:
    def __init__(self, test=False, description=None):
        self.test = test
        self.run = neptune.init_run(
            project=os.environ["NEPTUNE_PROJECT_NAME"],
            mode="offline" if test else "async",
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            description=description
        )

    def log_metadata(self, metadata):
        self.run["parameters"] = metadata

    def log_config_dict(self, config_dict, folder="config"):
        for key, value in config_dict.items():
            self.run[f"{folder}/{key}"] = value

    def log_metric(self, metric_name, metric_value, step=None):
        """
        Log a metric to Neptune with an optional step.

        Parameters:
            metric_name (str): The name of the metric to log.
            metric_value (float): The value of the metric.
            step (int, optional): The step to associate with the metric. Defaults to None.
        """
        if step is not None:
            self.run[metric_name].append(value=metric_value, step=step)
        else:
            self.run[metric_name].append(metric_value)
        if self.test:
            print(f"Logged {metric_name} with value {metric_value} at step {step}")

    def log_model(self, file_path, file_name):
        self.run[file_name].upload(file_path)

    def stop(self):
        self.run.stop()

    # ────────────────────────────────────────────────────────────────
    # Images / figures
    # ────────────────────────────────────────────────────────────────
    def log_image(self, series_name: str, file_path: str, step: int | None = None):
        """
        Append an image (PNG/JPG) to a Neptune series.

        Parameters
        ----------
        series_name : str   Neptune field, e.g. "reconstructions"
        file_path   : str   Path to image on disk
        step        : int   Optional x-axis step
        """
        from neptune.types import File
        if step is not None:
            self.run[series_name].append(File.as_image(file_path), step=step)
        else:
            self.run[series_name].append(File.as_image(file_path))
        if self.test:
            print(f"Logged image {file_path} → {series_name} at step {step}")

if __name__ == "__main__":
    import os
    logger = NeptuneLogger(
        test=False,
        description="Test run"
    )
    logger.run["metric"].append(1.0)
    logger.run["metric"].append(0.5)
    logger.run.stop()
