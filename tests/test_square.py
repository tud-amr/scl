from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader, random_split
from project.datatools.trajpred_dataset import TrajpredDataset
from project.training_routines.ewc import EwcPredictor
from project.models import *


def test_ewc_predictor():
    seed_everything(1234)

    # ------------
    # Model
    # ------------
    model = EwcPredictor(
        model='EthPredictor',
        target_mode='position'
    )

    # ------------
    # data
    # ------------
    dataset = TrajpredDataset(
        model.predictor,
        experiments=['square_6-agents'],
        stride=5,
        train=True
    )
    val_split = 0.2
    dataset_train, dataset_val = random_split(
        dataset,
        [len(dataset) - int(len(dataset) * val_split), int(len(dataset) * val_split)],
        generator=torch.Generator().manual_seed(1)
    )
    train_loader = DataLoader(dataset_train, batch_size=20, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset_val, batch_size=20, shuffle=True, num_workers=6)

    # ------------
    # training
    # ------------
    # state_dict = torch.load('saves/StateDiffs/coop_dev/final.ckpt')['state_dict']
    # model.load_state_dict(state_dict)
    trainer = Trainer(
        checkpoint_callback=False,
        logger=False,
        max_epochs=20,
        gradient_clip_val=0.5
    )
    trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, val_loader)
    print(result[0])

    assert result[0]['test_fde'] < 0.5


if __name__ == '__main__':
    test_ewc_predictor()
