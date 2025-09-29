from datasets import load_dataset

ds = load_dataset("Cancer-Myth/Cancer-Myth")

# save locally
ds.save_to_disk("cancer_myth_dataset")