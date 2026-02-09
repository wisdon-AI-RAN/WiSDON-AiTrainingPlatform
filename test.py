model_version = "1.0.0"
new_model_version = ".".join(model_version.split(".")[:-1] + [str(int(model_version.split(".")[-1]) + 1)])
print(model_version)
print(type(new_model_version))
print(model_version.split(".")[:-1] + [str(int(model_version.split(".")[-1]) + 1)])