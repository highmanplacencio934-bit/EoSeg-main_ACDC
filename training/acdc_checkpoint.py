"""Keep ACDC best selection intact while saving the latest evaluated epoch."""
from lightning.pytorch.callbacks import ModelCheckpoint


class ACDCModelCheckpoint(ModelCheckpoint):
    """Work around Lightning versions that refresh last only on a top-k save."""

    def _save_rolling_last(self, trainer, *, at_train_epoch_end):
        if (not self.save_last or self._should_skip_saving_checkpoint(trainer)
                or self._should_save_on_train_epoch_end(trainer) != at_train_epoch_end
                or self._every_n_epochs < 1
                or (trainer.current_epoch + 1) % self._every_n_epochs):
            return
        self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        self._save_rolling_last(trainer, at_train_epoch_end=False)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        self._save_rolling_last(trainer, at_train_epoch_end=True)
