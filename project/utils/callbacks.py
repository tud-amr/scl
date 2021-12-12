from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import pandas as pd


class TestEveryEpoch(Callback):

    def __init__(self, test_loaders):
        self.test_loaders = test_loaders
        self.current_task = 1
        self.current_method = 'vanilla'
        self.results = pd.DataFrame({'epoch': [], 'scenario': [], 'ade': [], 'fde': [], 'task': [], 'method': []})

        # Hack to have disable progress bar only for testing
        self._private_test_trainer = pl.Trainer(checkpoint_callback=False, progress_bar_refresh_rate=0)

    def on_train_epoch_start(self, trainer, pl_module):
        for i, t in enumerate(self.test_loaders):
            if i + 1 <= self.current_task:
                r = self._private_test_trainer.test(pl_module, t, verbose=False)
                self.results = self.results.append({'epoch': trainer.current_epoch, 'scenario': i, 'ade': r[0]['test_ade'], 'fde': r[0]['test_fde'], 'task': self.current_task, 'method': self.current_method}, ignore_index=True)
