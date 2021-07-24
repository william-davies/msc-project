# Save a model file from the current directory
wandb.save("model.h5")

# Save all files that currently exist containing the substring "ckpt"
wandb.save("../logs/*ckpt*")

# Save any files starting with "checkpoint" as they're written to
wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
