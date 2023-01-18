from .register import Register

# engine
LOOPS = Register('loops')
HOOKS = Register('hooks')
OPTIMIZERS = Register('optimizers')
SCHEDULERS = Register('schedulers')
EVALUATORS = Register('evaluators')

# dataset
DATASETS = Register('datasets')
TRANSFORMS = Register('transforms')
BATCH_SAMPLERS = Register('batch_sampler')

MODELS = Register('models')
