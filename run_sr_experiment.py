#!/usr/bin/env python
import os
import datetime
import multiprocessing


DATASET_DIR = "chesapeake_data/"
OUTPUT_DIR = "results/results_sr_epochs_100_0/"

_GPU_IDS = [0, 1, 2]
NUM_GPUS = len(_GPU_IDS)
JOBS_PER_GPU = [[] for i in range(NUM_GPUS)]

# pylint: disable=redefined-outer-name
def run_jobs(jobs):
    print("Starting job runner")
    for (command, args) in jobs:
        print(datetime.datetime.now(), command)

        output_dir = os.path.join(args["output"], args["exp_name"])
        os.makedirs(output_dir, exist_ok=True)
        os.system(command + " > %s 2>&1" % (os.path.join(output_dir, args["log_name"])))

        # process = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE,
        #           stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
        # with open(os.path.join(output_dir, args["log_name"]), 'w') as f:
        #     while process.returncode is None:
        #         for line in process.stdout:
        #             f.write(line.decode('utf-8').strip() + "\n")
        #         process.poll()


TRAIN_STATE_LIST = [
    "de_1m_2013",
    "ny_1m_2013",
    "md_1m_2013",
    "pa_1m_2013",
    "va_1m_2014",
    "wv_1m_2014",
]
TEST_STATE_LIST = [
    "de_1m_2013",
    "ny_1m_2013",
    "md_1m_2013",
    "pa_1m_2013",
    "va_1m_2014",
    "wv_1m_2014",
]

GPU_IDX = 0

for train_state in TRAIN_STATE_LIST:
    for test_state in TEST_STATE_LIST:

        if not os.path.exists(
            os.path.join(
                OUTPUT_DIR,
                "train-hr_%s_train-sr_%s/final_model.h5" % (train_state, test_state),
            )
        ):
            GPU_ID = _GPU_IDS[GPU_IDX]

            args = {
                "output": OUTPUT_DIR,
                "exp_name": "train-hr_%s_train-sr_%s" % (train_state, test_state),
                "TRAIN_STATE_LIST": train_state,
                "val_state_list": train_state,
                "superres_state_list": test_state,
                "gpu": GPU_ID,
                "data_dir": DATASET_DIR,
                "log_name": "log.txt",
                "learning_rate": 0.001,
                "loss": "superres",
                "batch_size": 16,
                "model_type": "unet_large",
            }

            COMMAND_TRAIN = (
                "python landcover/train_model_landcover.py "
                "--output {output} "
                "--name {exp_name} "
                "--gpu {gpu} "
                "--verbose 2 "
                "--data_dir {data_dir} "
                "--training_states {TRAIN_STATE_LIST} "
                "--validation_states {val_state_list} "
                "--superres_states {superres_state_list} "
                "--model_type {model_type} "
                "--learning_rate {learning_rate} "
                "--loss {loss} "
                "--batch_size {batch_size} "
            ).format(**args)
            JOBS_PER_GPU[GPU_IDX].append((COMMAND_TRAIN, args))

            args = {
                "test_csv": "{}/{}_extended-test_tiles.csv".format(
                    DATASET_DIR, test_state
                ),
                "output": "{}/train-hr_{}_train-sr_{}/".format(
                    OUTPUT_DIR, train_state, test_state
                ),
                "exp_name": "test-output_{}".format(test_state),
                "gpu": GPU_ID,
                "log_name": "log_test_{}.txt".format(test_state),
            }
            COMMAND_TEST = (
                "python landcover/testing_model_landcover.py "
                "--input {test_csv} "
                "--output {output}/{exp_name}/ "
                "--model {output}/final_model.h5 "
                "--gpu {gpu} "
                "--superres"
            ).format(**args)
            JOBS_PER_GPU[GPU_IDX].append((COMMAND_TEST, args))

            args = args.copy()
            args["log_name"] = "log_acc_{}.txt".format(test_state)
            COMMAND_ACC = (
                "python compute_accuracy.py "
                "--input {test_csv} "
                "--output {output}/{exp_name}/"
            ).format(**args)
            JOBS_PER_GPU[GPU_IDX].append((COMMAND_ACC, args))

            GPU_IDX = (GPU_IDX + 1) % NUM_GPUS
        else:
            print("Skipping %s-%s" % (train_state, test_state))


POOL_SZ = NUM_GPUS
POOL = multiprocessing.Pool(NUM_GPUS + 1)
POOL.map(run_jobs, JOBS_PER_GPU)
POOL.close()
POOL.join()
