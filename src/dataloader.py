import kagglehub

# Download latest version
path = kagglehub.dataset_download("balakrishcodes/brain-2d-mri-imgs-and-mask")

print("Path to dataset files:", path)