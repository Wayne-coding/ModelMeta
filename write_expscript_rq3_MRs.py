import os
import shlex
import subprocess

exp = "rq3_MRs"
models = ["SSDresnet50fpn","SSDmobilenetv1","resnet","TextCNN","openpose","crnn","unet","DeepLabV3","ssimae","vit"]
mutate_times = 100
ifeplison = 0.6
ifapimut = False
ifTompson = True
num_samples = 1
run_option = 2
# MRs = ["1","2","3","0,2,3", "1,2,3", "0,1,3", "0,1,2"]
MRs = ["0,1,2,3"]
num_quantiles = 20
device = 6
loss_name = "CrossEntropy"
opt_name = "SGD"
if not os.path.exists(f"./configs/{exp}"):
    os.mkdir(f"./configs/{exp}")
for MR in MRs:
    for model_name in models:
        yaml_path = f"/home/cvgroup/myz/czx/semtest-gitee/modelmeta/configs/{exp}/{model_name}_{ifapimut}.yaml"
        f = open(yaml_path,"w")
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
        
        f.write("train_config: \n")
        f.write(f"  loss_name: {loss_name}\n")
        f.write(f"  opt_name: {opt_name}\n")
        f.write(f"  seed_model: {model_name}\n")

        f.close()
        csv_path = "/home/cvgroup/myz/czx/semtest-gitee/modelmeta/rq3_gpu_mrs_dqn" + f"/{MR}_{model_name}_{ifapimut}.csv"
        main_yaml_path = f"/home/cvgroup/myz/czx/semtest-gitee/modelmeta/configs/mian.yaml"
        f = open(main_yaml_path,"w")
        f.write("config:\n")
        f.write(f"  yaml_path: {yaml_path}\n")
        f.write(f"  csv_path: {csv_path}\n")

        f.close()

        run_command = "coverage run --source=mindspore ./mindspore_mutation/main_ms.py"
        run_args = shlex.split(run_command)
        run_p = subprocess.Popen(run_args)
        run_p.communicate()
        # Now run coverage combine
        # combine_command = "coverage combine"
        # combine_args = shlex.split(combine_command)
        # combine_p = subprocess.Popen(combine_args)
        # combine_p.communicate()

        # Then run coverage html
        html_command = "coverage html"
        os.system(html_command)


        run_command = "/home/cvgroup/miniconda3/envs/czx/bin/python ./caculation.py"
        html_args = shlex.split(run_command)
        html_p = subprocess.Popen(html_args)
        html_p.communicate()

 