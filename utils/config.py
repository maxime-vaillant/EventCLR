from typing import List, Tuple

from datasets.dataset_factory import DatasetFactory


class ExperimentConfig:
    """Configuration for experiments."""

    def __init__(
        self,
        # Experiment type
        experiment_type: str = 'supervised',  # 'supervised' or 'self_supervised'

        # Datasets
        target_dataset: str = 'cifar10dvs',
        ssl_datasets: List[str] = None,  # For self-supervised pretraining

        # Architecture
        backbone_name: str = 'resnet18',
        backbone_width: float = 1.0,
        projection_dim: int = 128,
        neuron_type: str = 'LIF',  # 'LIF', 'IF', 'PLIF', or 'ReLU' for the ANN-equivalent baseline

        # Training parameters
        pretrain_epochs: int = 500,
        eval_epochs: int = 150,
        pretrain_batch_size: int = 256,
        eval_batch_size: int = 64,
        pretrain_lr: float = 1e-3,
        eval_lr_linear: float = 1e-3,
        eval_lr_finetune: float = 1e-3,
        pretrain_weight_decay: float = 1e-4,
        eval_weight_decay: float = 1e-4,

        # Data parameters
        n_time_bins: int = 4,
        resize_size: Tuple[int, int] = (48, 48),
        representation: str = 'frame',  # 'frame' or 'voxel'
        normalize: bool = True,
        samples_per_class_list: List[float] = None,
        samples_mode: str = 'percent',  # 'percent' or 'count'
        n_subset_runs: int = 3,
        pretrain_samples_percentage: float = 1.0,  # Percentage of pretraining data to use

        # CutMix parameters
        use_cutmix: bool = True,
        cutmix_prob: float = 0.5,

        # Hardware
        device: int = 0,
        num_workers: int = 8,

        # Other
        seed: int = 42,

        # Augmentation setup
        aug_setup: str = 'eventclr',  # 'eventclr', 'nda', or 'eventdrop'

        # Pretrained weights
        pretrained_run_id: str = None,  # MLflow run ID to load pretrained weights from

        # Temporal loss strategy
        pretrain_loss_strategy: str = 'naive',  # 'naive' or 'temporal'
        supervised_loss_strategy: str = 'naive',  # 'naive' or 'temporal'

        # Evaluation types to run
        eval_types: List[str] = None,  # subset of ['lp', 'ft', 'sup']

        # Run name
        run_name: str = None,

        # Augmentation families to enable (for ablation studies)
        aug_families: List[str] = None,  # subset of ['temporal', 'spatial', 'polarity']

        **neuron_kwargs
    ):
        self.experiment_type = experiment_type

        self.target_dataset = target_dataset
        self.ssl_datasets = ssl_datasets or [target_dataset]

        self.backbone_name = backbone_name
        self.backbone_width = backbone_width
        self.projection_dim = projection_dim

        self.pretrain_epochs = pretrain_epochs
        self.eval_epochs = eval_epochs
        self.pretrain_batch_size = pretrain_batch_size
        self.eval_batch_size = eval_batch_size
        self.pretrain_lr = pretrain_lr
        self.eval_lr_linear = eval_lr_linear
        self.eval_lr_finetune = eval_lr_finetune
        self.pretrain_weight_decay = pretrain_weight_decay
        self.eval_weight_decay = eval_weight_decay

        self.n_time_bins = n_time_bins
        self.resize_size = resize_size
        self.representation = representation
        self.normalize = normalize
        self.samples_per_class_list = samples_per_class_list
        self.samples_mode = samples_mode
        self.n_subset_runs = n_subset_runs
        self.pretrain_samples_percentage = pretrain_samples_percentage

        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob

        self.device = device
        self.num_workers = num_workers
        self.seed = seed
        self.aug_setup = aug_setup
        self.pretrained_run_id = pretrained_run_id

        self.pretrain_loss_strategy = pretrain_loss_strategy
        self.supervised_loss_strategy = supervised_loss_strategy

        self.eval_types = eval_types if eval_types is not None else ['lp', 'ft', 'sup']
        self.run_name = run_name
        self.aug_families = frozenset(aug_families) if aug_families is not None else frozenset(['temporal', 'spatial', 'polarity'])

        # Dataset-specific attributes for target dataset
        self.sensor_size = DatasetFactory.get_sensor_size(target_dataset)
        self.num_classes = DatasetFactory.get_num_classes(target_dataset)

        self.neuron_type = neuron_type
        self.cnf = 'ADD'
        self.neuron_kwargs = neuron_kwargs