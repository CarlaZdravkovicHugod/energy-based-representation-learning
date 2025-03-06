"""
Neptune Logger for logging metrics, models, and metadata to Neptune.
"""

import neptune
import os

class NeptuneLogger:
    def __init__(self, test=False):
        self.test = test
        self.run = neptune.init_run(
            project=os.environ["NEPTUNE_PROJECT_NAME"],
            mode="offline" if test else "async",
            api_token=os.environ["NEPTUNE_API_TOKEN"],
        )

    def log_metadata(self, metadata):
        self.run["parameters"] = metadata

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

if __name__ == "__main__":
    import os
    logger = NeptuneLogger(
        test=False
    )
    logger.run["metric"].append(1.0)
    logger.run["metric"].append(0.5)
    logger.run.stop()
