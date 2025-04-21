ColorMamba/
├── assets/                     # Images and pre-trained models
│   ├── autoencoder.pth        # Pre-trained autoencoder
│   └── colormamba_architecture.png
├── datasets/                   # Dataset directory (to be populated)
│   ├── NIR/                   # NIR images
│   ├── RGB-Registered/        # RGB images
│   ├── Validation/            # Validation images
│   └── Testing/               # Testing images
├── models/                     # Model definitions
│   ├── __init__.py
│   ├── agent_att.py           # Agent-based attention
│   ├── commom.py              # Common modules (SPADE, AutoEncoder)
│   ├── CycleGanNIR_net.py     # Main generator and discriminator
│   ├── gen_net.py             # HSV prediction network
│   ├── mambair_arch.py        # VSSB and MambaIR implementation
│   ├── spectral_normalization.py # Spectral normalization
│   └── trans.py               # Criss-cross attention
├── tools/                      # Utility functions
│   ├── __init__.py
│   ├── fit.py                 # Training logic
│   ├── losses.py              # Loss functions
│   ├── tools.py               # Evaluation metrics
│   └── utils.py               # Miscellaneous utilities
├── data_loader.py             # Dataloader for NIR-RGB pairs
├── main.py                    # Placeholder script
├── test.py                    # Testing script
├── train.py                   # Training script
├── requirements.txt           # Dependencies (to be created)
└── README.md                  # This file
