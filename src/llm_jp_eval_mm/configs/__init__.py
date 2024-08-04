import os

from omegaconf import OmegaConf

file_path = os.path.abspath(__file__)
configs_dir = os.path.dirname(file_path)

print(f"Loading configs from {configs_dir} !")

config_names = os.getenv("CONFIG", None)
if config_names is None:
    config_names = "base_config"  # Modify this if you want to use another default config

configs = [OmegaConf.load(os.path.join(configs_dir, "base_config.yaml"))]

if config_names is not None:
    for config_name in config_names.split(","):
        if os.path.exists(os.path.join(configs_dir, f"{config_name}.yaml")):
            print(f"Loading config: {config_name}")
            configs.append(OmegaConf.load(os.path.join(configs_dir, f"{config_name}.yaml")))

# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)

print(f"Loaded configs: {OmegaConf.to_yaml(config)}")
