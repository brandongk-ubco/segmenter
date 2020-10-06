from .IsCompleteJob import IsCompleteJob
from .TrainJob import TrainJob
from .BaseJob import BaseJob
from .GridSearchJob import GridSearchJob
from .TrainAllJob import TrainAllJob

tasks = [TrainJob, IsCompleteJob, GridSearchJob, TrainAllJob]
