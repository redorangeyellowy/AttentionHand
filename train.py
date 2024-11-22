import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.hack import disable_verbosity
disable_verbosity()

# Configs
resume_path = 'weights/sd15_ini.ckpt'
batch_size = 1
logger_freq = 100
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/attentionhand.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
