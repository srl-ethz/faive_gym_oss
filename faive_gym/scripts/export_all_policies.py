'''
Calls export_policy.py in a loop to export all policies that are in the models/ directory (beware: not the default
 saving dir to avoid saving all checkpoints) in the .onnx (ONNX) and .pt
(TorchScript module) format. The output filename will be in the <policy_name>_YYYY-MM-DD_HH-mm-ss.pth/.onnx format

specify task name as argument, e.g.:
python export_all_policies.py FaiveHandP0_sphere
'''

import subprocess
import glob
import os
import sys
MODEL_FOLDER="../models"

if len(sys.argv) < 2:
    print("Please specify the task name as argument, e.g.:")
    print("python export_all_policies.py FaiveHandP0_sphere")
    exit()
TASK_NAME=sys.argv[1]

# iterate through MODEL_FOLDER and export all .pth files as
# .onnx and .pt
for policy_file in glob.glob(os.path.join(MODEL_FOLDER, "*.pth")):
    name = os.path.splitext(os.path.basename(policy_file))[0]
    print("Found policy file ", policy_file, " exporting with name ", name, "...")
    command=f"python export_policy.py task={TASK_NAME} checkpoint={policy_file} wandb_name={name}"
    subprocess.run(command, shell=True)
    print("Successfully exported ", policy_file)