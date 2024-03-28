import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig

configs = []

for lr in np.logspace(-4, -2, 10):
    config = TrainConfig(
        data=DataConfig(
            builder=FunctionConfig(
                name='zoology.data.associative_recall.multiquery_ar',
                kwargs={'num_kv_pairs': 4, 'train_power_a': 0.01, 'test_power_a': 0.01, 'random_non_queries': False}
            ),
            seed=0,
            num_train_examples=100000,
            num_test_examples=3000,
            input_seq_len=64,
            vocab_size=8192,
            batch_size=512,
            caching=True,
            force_cache=False
        ),
        model=ModelConfig(
            sequence_mixer=ModuleConfig(name='zoology.mixers.rwkv6.RWKV_Tmix_x060', kwargs={'l_max': 64}),
            state_mixer=ModuleConfig(name='torch.nn.Identity', kwargs={}),
            d_model=64,
            n_layers=4,
            max_position_embeddings=0,
            learnable_word_embeddings=True,
            vocab_size=8192,
            resid_dropout=0.0,
            embed_dropout=0.1,
            drop_path=0.0,
            layer_norm_epsilon=1e-05,
            pad_vocab_size_multiple=1,
            block_type='TransformerBlock'
        ),
        logger=LoggerConfig(project_name='zoology_sweep_testing', entity='gpt6'),
        max_epochs=64,
        early_stopping_metric='valid/accuracy',
        early_stopping_threshold=0.99,
        learning_rate=lr,
        weight_decay=0.1,
        seed=123,
    )
    configs.append(config)