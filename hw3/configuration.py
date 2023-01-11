class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr_backbone = 1e-5
        # self.lr = 1e-4
        self.lr = 8e-5

        # Epochs
        self.epochs = 30
        self.lr_drop = 20
        self.start_epoch = 0
        # self.weight_decay = 1e-4
        self.weight_decay = 1e-5

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 8
        self.num_workers = 4
        # self.checkpoint = './checkpoint_pretrained.pth'
        self.checkpoint = './27_model_best_225.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.225
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = '../hw3_data/p2_data'
        self.limit = -1
