import os

exp = "rq3_IfAPI"
models = ["vgg16"]
mutate_times = 10
ifeplison = 0
ifapimuts = [True,False]
ifTompson = True
num_samples = 1
run_option = 2
MR = "0,1,2,3"
num_quantiles = 20
device = 1

if not os.path.exists(f"./configs/{exp}"):
    os.mkdir(f"./configs/{exp}")

for ifapimut in ifapimuts:
    for model_name in models:
        f = open(f"./configs/{exp}/{model_name}_{ifapimut}.yaml","w")
        f.write("execution_config:\n")
        f.write(f"  seed_model: {model_name}\n")
        f.write(f"  mutate_times: {mutate_times}\n")
        f.write(f"  ifeplison: {ifeplison}\n")
        f.write(f"  ifapimut: {ifapimut}\n")
        f.write(f"  ifTompson: {ifTompson}\n")
        f.write(f"  num_samples: {num_samples}\n")
        f.write(f"  run_option: {run_option}\n")
        f.write(f"  MR: {MR}\n")
        f.write(f"  num_quantiles: {num_quantiles}\n")
        f.write(f"  device: {device}\n")

        f.close()