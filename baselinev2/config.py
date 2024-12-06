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
    # PREPROCESSED_DATA_ROOT = '/data/yaz/birdclef24/data/specs'
    # LOAD_DATA = True  # whether to load data from pre-processed dataset

    image_size = 256

    SR = 32000  # sample rate
    mel_spec_params = {
        "sample_rate": 32000,
        "n_mels": 128,
        "f_min": 20,
        "f_max": 16000,
        "n_fft": 2048,
        "hop_length": 512,
        "normalized": True,
        "center" : True,
        "pad_mode" : "constant",
        "norm" : "slaney",
        "onesided" : True,
        "mel_scale" : "slaney"
    }

    top_db = 80 

    train_period = 5
    val_period = 5
    secondary_coef = 1.0
    train_duration = train_period * mel_spec_params["sample_rate"]
    val_duration = val_period * mel_spec_params["sample_rate"]

    # == model config ==
    MODEL_TYPE = 'efficientnet_b0'  # model type
    
    # == dataset config ==
    # BATCH_SIZE = 128  # batch size of each step
    BATCH_SIZE = 256 # batch size of each step
    N_WORKERS = 4  # number of workers
    
    # == AUG ==
    USE_XYMASKING = True  # whether use XYMasking
    
    # == training config ==
    N_FOLDS = 4  # n fold
    EPOCHS = 30  # max epochs
    LR = 1e-3  # learning rate
    WEIGHT_DECAY = 1e-5  # weight decay of optimizer
    
    # == other config ==
    VISUALIZE = False # whether to visualize data and batch