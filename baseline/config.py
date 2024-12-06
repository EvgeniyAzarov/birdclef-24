class Config:
    # == global config ==
    SEED = 42 # random seed
    DEVICE = 'cuda'  # device to be used
    MIXED_PRECISION = False  # whether to use mixed-16 precision
    # OUTPUT_DIR = '/kaggle/working/'  # output folder
    OUTPUT_DIR = '/data/yaz/birdclef24/out/'  # output folder
    
    # == data config ==
    # DATA_ROOT = '/kaggle/input/birdclef-2024'  # root folder
    DATA_ROOT = '/data/yaz/birdclef24/data'  # root folder
    # PREPROCESSED_DATA_ROOT = '/kaggle/input/birdclef24-spectrograms-via-cupy'
    PREPROCESSED_DATA_ROOT = '/data/yaz/birdclef24/data/specs'
    LOAD_DATA = True  # whether to load data from pre-processed dataset
    FS = 32000  # sample rate
    N_FFT = 1095  # n FFT of Spec.
    WIN_SIZE = 412  # WIN_SIZE of Spec.
    WIN_LAP = 100  # overlap of Spec.
    MIN_FREQ = 40  # min frequency
    MAX_FREQ = 15000  # max frequency
    
    # == model config ==
    # MODEL_TYPE = 'efficientnet_b0'  # model type
    
    # == dataset config ==
    BATCH_SIZE = 128  # batch size of each step
    N_WORKERS = 4  # number of workers
    
    # == AUG ==
    USE_XYMASKING = True  # whether use XYMasking
    
    # == training config ==
    FOLDS = 4  # n fold
    EPOCHS = 15  # max epochs
    LR = 1e-3  # learning rate
    WEIGHT_DECAY = 1e-5  # weight decay of optimizer
    
    # == other config ==
    VISUALIZE = True  # whether to visualize data and batch