import pandas as pd
import asyncio
import datetime
from dateutil import parser
from dataclasses import asdict
from neuromation.cli.rc import ConfigFactory, Config
from neuromation.client.jobs import JobStatus, JobDescription
from pprint import pprint
from typing import List, Dict, Union, Any
import itertools
from pathlib import Path
import subprocess

from tqdm import tqdm


COLUMNS = [
    "N_EPOCHS",
    "TRAIN_DATASETS",
    "TEST_DATASETS",
    "NORMALIZE_EMBEDDINGS",
    "SHARE_MODELS",
    "SUCCESSIVE_ATTENTION",
    "SPARSE_EMBEDDINGS",
    "SHARED_ATTENTION",
    "CONSTANT_EMBEDDINGS",
    "SINGLE_INPUT",
    "MULTIPLE_CHOICE_HEAD",
    "DIALOG_EMBEDDINGS",
    "USE_START_END",
    "PERSONA_AUGMENT",
    "PERSONA_AUG_SYN_PROBA",
    "S2S_WEIGHT",
    "LM_WEIGHT",
    "RISK_WEIGHT",
    "HITS_WEIGHT",
    "NEGATIVE_SAMPLES",
    "BEAM_SIZE",
    "DIVERSITY_COEF",
    "DIVERSITY_GROUP",
    "ANNEALING_TOPK",
    "ANNEALING",
    "LENGTH_PENALTY",
    "TRAIN_BATCH_SIZE",
    "BATCH_SPLIT",
    "TEST_BATCH_SIZE",
    "LABEL_SMOOTHING",
    "FP16",
    "LOSS_SCALE",
    "LINEAR_SCHEDULE",
    "EVALUATE_FULL_SEQUENCES",
    "LIMIT_EVAL_TIME",
    "LIMIT_TRAIN_TIME",
]


async def get_jobs_info(
        cfg: Config, job_ids: List[str]) -> Dict[str, JobDescription]:
    job_info = {}
    async with cfg.make_client() as client:
        for job in job_ids:
            try:
                status = await client.jobs.status(job)
                job_info[job] = status
            except ValueError as e:
                print(e)
    return job_info


def convert_timedelta(duration: datetime.timedelta) -> str:
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"


def create_experiments_report(jobs: List[Dict[str, Union[str, int]]],
                              path_to_save: Path):
    loop = asyncio.get_event_loop()
    config = ConfigFactory.load()
    jobs_info = loop.run_until_complete(
        get_jobs_info(config, [x['id'] for x in jobs]))

    for job in jobs:
        info = jobs_info[job['id']]
        info = asdict(info)

        container = info['container']
        env = container['env']
        for env_var in COLUMNS:
            job[env_var] = env.get(env_var, 'NA/Default')

        job['image'] = container['image']
        resources = container['resources']
        del resources['shm']
        job.update(resources)

        history = info['history']
        status = history['status']
        del history['status']
        del history['reason']
        del history['description']
        del history['created_at']
        job['status'] = status
        if status == JobStatus.RUNNING:
            print("Job is running")
            history['total_time'] = 'NA'
        else:
            history['total_time'] = convert_timedelta((parser.parse(
                history['finished_at']) - parser.parse(history['started_at'])))

        job.update(history)
        job['description'] = info['description']
        job['is_preemptible'] = info['is_preemptible']

    df = pd.DataFrame(jobs)
    print(len(df.columns))
    columns = ['id',
               'is_preemptible',
               'status',
               'description',
               'started_at',
               'finished_at',
               'total_time',
               ] + COLUMNS + ['cpu',
                              'gpu',
                              'gpu_model',
                              'image',
                              'memory_mb',
                              'batch']
    print(len(columns))
    df = df[columns]
    df.to_csv(path_to_save, index=False)


def find_logs_path(job_id: str, runs_path: Path) -> Path:
    logs = [log for log in runs_path.iterdir()
            if job_id in log.name and (log / 'eval_references_file').exists()]
    assert len(logs) == 1
    return logs[0]


def run_compare_mt(jobs: List[Dict[str, Union[str, int]]], runs_path: Path):
    # filter by batch 1, because we did't save eval_predictions_file
    # and eval_references_file for batch 1
    jobs = [x for x in jobs if x['batch'] != 1]
    for job in jobs:
        job['logs'] = find_logs_path(job['id'], runs_path=runs_path)

    # for each 2 pairs
    for (job1, job2) in tqdm(itertools.combinations(jobs, 2)):
        ref = job1['logs'] / 'eval_references_file'
        sample1 = job1['logs'] / 'eval_predictions_file'

        sample2 = job2['logs'] / 'eval_predictions_file'
        job1_id = job1['id']
        job2_id = job2['id']

        cmd = ['compare-mt', f'{ref}', f'{sample1}', f'{sample2}', '--output_directory', f'compare-mt-results/res-{job1_id}_vs_{job2_id}', '--sys_names', job1['id'], job2['id']]
        print(" ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        _ = proc.communicate()
        print('code: ' + str(proc.returncode))

    # for each all
    ref =  jobs[0]['logs'] / 'eval_references_file'

    cmd = ['compare-mt', f'{ref}'] + [str(j['logs'] / 'eval_predictions_file') for j in jobs] + ['--output_directory', f'compare-mt-results/all-vs-all', '--sys_names'] + [str(j['id']) for j in jobs]
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    _ = proc.communicate()
    print('code: ' + str(proc.returncode))



if __name__ == '__main__':
    jobs = [
        {'id': 'job-08b07f85-61fd-49fd-abd0-659f2dfd2788', 'batch': 1},
        {'id': 'job-d0b1a8f5-bf2d-4da5-a7fb-50e93381ca03', 'batch': 1},
        {'id': 'job-53bb3691-153c-4ba1-96e7-2d3e6e1a1c4e', 'batch': 1},
        {'id': 'job-8e66f304-2537-4e73-83e8-af3215e440b1', 'batch': 1},
        {'id': 'job-63e31577-becb-4f1d-961d-27f41705418d', 'batch': 1},
        {'id': 'job-bbafbfa5-140f-4f4a-bbd7-1259521fecce', 'batch': 1},

        {'id': 'job-8a2b2a96-e739-4440-9b53-520887a47252', 'batch': 2},
        {'id': 'job-685c3437-422e-411c-b084-18c8628eaaf5', 'batch': 2},
        {'id': 'job-8c4cec8e-0b33-44ec-aee6-20ef8b146345', 'batch': 2},
        {'id': 'job-6e04c126-d0b7-440f-818f-257ceb80de8d', 'batch': 2},
        {'id': 'job-f0c37e89-462b-4c04-ae04-ac2a73bd6712', 'batch': 2},
        {'id': 'job-a2d20720-0503-451d-8496-f2d5a6b66a29', 'batch': 2},
        {'id': 'job-b54ac8a8-e7ce-443b-a388-edaaa9b8528d', 'batch': 2},
        {'id': 'job-817fe109-2e68-4fff-9ea6-c9a463879faa', 'batch': 2},
        {'id': 'job-c212c9ad-2980-40c9-92b5-09a1ba8d8db9', 'batch': 2},
        {'id': 'job-193e7eb0-9ae0-46e9-9740-c6d0de6b63bd', 'batch': 2},
        {'id': 'job-aea1a6d8-654c-4c1b-853f-6c4ba215a549', 'batch': 2},
        {'id': 'job-d5ad1c7c-b430-4186-8ff5-aaf94d5f0942', 'batch': 2},
        {'id': 'job-a1055649-13b4-4778-8dfc-82aa0cbbc19b', 'batch': 2},
        {'id': 'job-72364ed9-3d3e-4cab-9d15-af5c5ca23e39', 'batch': 2},
        {'id': 'job-9eda87c5-9dd6-438d-a692-7a56963b3ee4', 'batch': 2},
        {'id': 'job-e490e6ec-7957-41f0-9c04-7acadb09dbb5', 'batch': 2},
        {'id': 'job-9f0b7bf9-7978-4bad-83b3-71c558b1fb24', 'batch': 2},

    ]
    # path_to_save = Path('df.csv')
    # create_experiments_report(jobs=jobs, path_to_save=path_to_save)

    runs_path = Path('/workspace/runs')
    run_compare_mt(jobs=jobs, runs_path=runs_path)
