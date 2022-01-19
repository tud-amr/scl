import random
from torch.utils.data import DataLoader
from project.training_routines.ewc import EwcPredictor
from project.datatools.trajpred_dataset import TrajpredDataset
from project.utils.metrics import ade, fde, cv_ade, cv_fde
from project.models import *
from tqdm import tqdm

# helper
def split_task(task, val_size):
    deck = list(range(len(task)))
    random.shuffle(deck)
    val_idxs = deck[:val_size]

    train_mask = np.ones(len(task), np.bool)
    train_mask[val_idxs] = 0
    task_train = deepcopy(task)
    for key, value in task_train._data.items():
        task_train._data[key] = value[train_mask]

    val_mask = np.zeros(len(task), np.bool)
    val_mask[val_idxs] = 1
    task_val = deepcopy(task)
    for key, value in task_val._data.items():
        task_val._data[key] = value[val_mask]

    return task_train, task_val

# ------------
# Experiment settings
# ------------
task_train_steps = 250
task_order = ['square','obstacle','hallway',]
batch_size = 20
val_size = 100

res_dir = 's-o-h'
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

results = []

# ------------
# Models
# ------------
model = 'StateDiffs'
save = 'eth-ucy_coreset'
# warmstart_ckpt = f'saves/{model}/{save}/epoch=15-step=1183-val_ade=0.26.ckpt'
# warmstart_ckpt = f'saves/{model}/{save}/epoch=13-step=2197-val_ade=0.22.ckpt'
warmstart_ckpt = f'saves/{model}/{save}/final.ckpt'
warmstart = torch.load(warmstart_ckpt) #['model']

""" """

model1 = EwcPredictor.load_from_checkpoint(warmstart_ckpt, ae_state_dict='saves/misc/eth_autoencoder.h5')
model1.ewc_weight = 0
optimizer1 = torch.optim.Adam(model1.predictor.parameters(), lr=1e-3, weight_decay=0)
model1.ol_step = warmstart['global_step']

model2 = EwcPredictor.load_from_checkpoint(warmstart_ckpt, ae_state_dict='saves/misc/eth_autoencoder.h5')
model2.ewc_weight = 1e6
optimizer2 = torch.optim.Adam(model2.predictor.parameters(), lr=1e-3, weight_decay=0)
model2.ol_step = warmstart['global_step']

model3 = EwcPredictor.load_from_checkpoint(warmstart_ckpt, ae_state_dict='saves/misc/eth_autoencoder.h5')
model3.ewc_weight = 1e6
optimizer3 = torch.optim.Adam(model3.predictor.parameters(), lr=1e-3, weight_decay=0)
model3.ol_step = warmstart['global_step']
model3.coreset_update_length = 20

model4 = EwcPredictor.load_from_checkpoint(warmstart_ckpt, ae_state_dict='saves/misc/eth_autoencoder.h5')
model4.ewc_weight = 0
optimizer4 = torch.optim.Adam(model4.predictor.parameters(), lr=1e-3, weight_decay=0)
model4.ol_step = warmstart['global_step']
model4.coreset_update_length = 20

# ------------
# data
# ------------
task_square = TrajpredDataset(
    model4.predictor,
    data_dir='data',
    train=True,
    experiments=['square_6-agents'],
    frequency=5,
    stride=10,
    tbptt=15,
    min_track_length=1.0,
)
task_square_train, task_square_val = split_task(task_square, val_size)
del task_square

task_obstacle = TrajpredDataset(
    model4.predictor,
    data_dir='data',
    train=True,
    experiments=['obstacles_4-agents'],
    frequency=5,
    stride=10,
    tbptt=15,
    min_track_length=1.0,
)
task_obstacle_train, task_obstacle_val = split_task(task_obstacle, val_size)
del task_obstacle

task_hallway = TrajpredDataset(
    model4.predictor,
    data_dir='data',
    train=True,
    experiments=['hallway_2-agents'],
    frequency=5,
    stride=6,
    tbptt=15,
    min_track_length=1.0,
)
task_hallway_train, task_hallway_val = split_task(task_hallway, val_size)
del task_hallway

# leave one out strategy
# task_unseen = TrajpredDataset(
#     model1.predictor,
#     data_dir='data',
#     train=True,
#     experiments=['unseen_4-agents'],
#     frequency=5,
#     stride=6,
#     tbptt=15,
#     min_track_length=1.0,
# )
# _, task_unseen_val = split_task(task_unseen, val_size)
# del task_unseen

tasks = {
    'square': {
        'train': DataLoader(task_square_train, batch_size=batch_size, shuffle=True, num_workers=6),
        'val': DataLoader(task_square_val, batch_size=1, shuffle=False, num_workers=6)
    },
    'obstacle': {
        'train': DataLoader(task_obstacle_train, batch_size=batch_size, shuffle=True, num_workers=6),
        'val': DataLoader(task_obstacle_val, batch_size=1, shuffle=False, num_workers=6)
    },
    'hallway': {
        'train': DataLoader(task_hallway_train, batch_size=batch_size, shuffle=True, num_workers=6),
        'val': DataLoader(task_hallway_val, batch_size=1, shuffle=False, num_workers=6)
    },
    # 'unseen': {
    #     'val': DataLoader(task_unseen_val, batch_size=1, shuffle=False, num_workers=6)
    # },
}

# ------------
# loops
# ------------
def val_loop(model, task, model_name, task_id):
    with torch.no_grad():
        for example_id, batch in enumerate(task):
            inputs, targets = batch
            preds = model.predictor(inputs)
            results.append({
                'model': model_name,
                'task': task_id,
                'step': model.ol_step,
                'example_id': example_id,
                'ade': float(ade(preds, targets)),
                'cv_ade': float(cv_ade(inputs, targets)),
                'cv_fde': float(cv_fde(inputs, targets)),
                'fde': float(fde(preds, targets))
            })

def train_loop(model_name, model, optimizer, train_task, val_tasks, max_steps):
    pbar = tqdm(total=max_steps-model.ol_step)
    while model.ol_step < max_steps:
        for batch in train_task:
            optimizer.zero_grad()
            inputs, targets = batch
            preds = model.predictor(inputs)
            batch_ade = ade(preds, targets)
            loss = batch_ade + model._compute_consolidation_loss()
            loss.backward()
            optimizer.step()
            model.ol_step += 1
            pbar.update(1)
            pbar.set_description(f'loss = {loss.item():.4f}')

            if model.ol_step % 50 == 0 and model.ol_step != 0:
                # val_loop(model, tasks['unseen']['val'], model_name, 4)
                for i, val_task in enumerate(val_tasks):
                    val_loop(model, val_task, model_name, i)

            if model.ol_step >= max_steps:
                break

model5 = EwcPredictor.load_from_checkpoint('saves/StateDiffs/offline_results/final.ckpt', ae_state_dict='saves/misc/eth_autoencoder.h5')
model5.ol_step = warmstart['global_step']
#
val_loop(model5, tasks['square']['val'], 'offline', 0)
val_loop(model5, tasks['obstacle']['val'], 'offline', 1)
val_loop(model5, tasks['hallway']['val'], 'offline', 2)
# ------------
# Model1
# ------------
""" """
val_tasks = []
for i, task_name in enumerate(task_order):
    if task_name not in val_tasks:
        val_tasks.append(task_name)
    train_loop(
        model_name='vanilla',
        model=model1,
        optimizer=optimizer1,
        train_task=tasks[task_name]['train'],
        val_tasks=[tasks[val_task_name]['val'] for val_task_name in val_tasks],
        max_steps=model1.ol_step+task_train_steps
    )
    torch.save(model1.state_dict(), f'{res_dir}/model1_task{i+1}.pt')

# ------------
# Model2
# ------------
val_tasks = []
for i, task_name in enumerate(task_order):
    if task_name not in val_tasks:
        val_tasks.append(task_name)
    train_loop(
        model_name='ewc',
        model=model2,
        optimizer=optimizer2,
        train_task=tasks[task_name]['train'],
        val_tasks=[tasks[val_task_name]['val'] for val_task_name in val_tasks],
        max_steps=model2.ol_step+task_train_steps
    )
    model2.register_ewc_params(tasks[task_name]['train'], i+1)
    torch.save(model2.state_dict(), f'{res_dir}/model2_task{i+1}.pt')

# ------------
# Model3
# ------------
val_tasks = []
for i, task_name in enumerate(task_order):
    if task_name not in val_tasks:
        val_tasks.append(task_name)
    train_loop(
        model_name='ewc_coreset',
        model=model3,
        optimizer=optimizer3,
        train_task=model3.add_coreset_to_loader(tasks[task_name]['train']),
        val_tasks=[tasks[val_task_name]['val'] for val_task_name in val_tasks],
        max_steps=model3.ol_step+task_train_steps
    )
    model3.register_ewc_params(tasks[task_name]['train'], i+1)
    model3.update_coreset(tasks[task_name]['train'])
    torch.save(model3.state_dict(), f'{res_dir}/model3_task{i+1}.pt')

val_tasks = []
for i, task_name in enumerate(task_order):
    if task_name not in val_tasks:
        val_tasks.append(task_name)
    train_loop(
        model_name='coreset',
        model=model4,
        optimizer=optimizer4,
        train_task=model4.add_coreset_to_loader(tasks[task_name]['train']),
        val_tasks=[tasks[val_task_name]['val'] for val_task_name in val_tasks],
        max_steps=model4.ol_step+task_train_steps
    )
    model4.register_ewc_params(tasks[task_name]['train'], i+1)
    model4.update_coreset(tasks[task_name]['train'])
    torch.save(model4.state_dict(), f'{res_dir}/model4_task{i+1}.pt')

# ------------
# Model3
# ------------
# train_loop(
#     model_name='ewc_coreset',
#     model=model3,
#     optimizer=optimizer3,
#     train_task=model3.add_coreset_to_loader(task1_train_loader),
#     val_tasks=[task1_val_loader],
#     max_steps=model3.ol_step+task_train_steps
# )
# model3.register_ewc_params(task1_train_loader, 1)
# model3.update_coreset(task1_train_loader)
# torch.save(model3.state_dict(), res_dir+'/model3_task1.pt')
#
# train_loop(
#     model_name='ewc_coreset',
#     model=model3,
#     optimizer=optimizer3,
#     train_task=model3.add_coreset_to_loader(task2_train_loader),
#     val_tasks=[task1_val_loader, task2_val_loader],
#     max_steps=model3.ol_step+task_train_steps
# )
# model3.register_ewc_params(task2_train_loader, 2)
# model3.update_coreset(task2_train_loader)
# torch.save(model3.state_dict(), res_dir+'/model3_task2.pt')
#
# train_loop(
#     model_name='ewc_coreset',
#     model=model3,
#     optimizer=optimizer3,
#     train_task=model3.add_coreset_to_loader(task3_train_loader),
#     val_tasks=[task1_val_loader, task2_val_loader, task3_val_loader],
#     max_steps=model3.ol_step+task_train_steps
# )
# model3.register_ewc_params(task3_train_loader, 3)
# model3.update_coreset(task3_train_loader)
# torch.save(model3.state_dict(), res_dir+'/model3_task3.pt')

# ------------
# Save
# ------------
import pandas as pd
res_df = pd.DataFrame(results)
res_df.to_csv(res_dir+'/results.csv')

print("Resultls saved in {}".format(res_dir+'/results.csv'))
