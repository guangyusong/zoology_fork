
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from zoology.data.ar_extrapolate import ARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "kvs-lin-attn-sweep" + sweep_id

VOCAB_SIZE = 8_192

train_configs = [
    # ARConfig(input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
    # ARConfig(input_seq_len=128, num_examples=20_000, num_kv_pairs=8, num_queries=8),
    # ARConfig(input_seq_len=256, num_examples=20_000, num_kv_pairs=16, num_queries=16),
    # ARConfig(input_seq_len=256, num_examples=20_000, num_kv_pairs=32, num_queries=32),
    # ARConfig(input_seq_len=256, num_examples=20_000, num_kv_pairs=64, num_queries=64),
    ARConfig(input_seq_len=64, num_examples=100_000, num_kv_pairs=4, num_queries=4),
    ARConfig(input_seq_len=64, num_examples=20_000, num_kv_pairs=8, num_queries=8),
    ARConfig(input_seq_len=64, num_examples=20_000, num_kv_pairs=16, num_queries=16),
    ARConfig(input_seq_len=128, num_examples=10_000, num_kv_pairs=32, num_queries=32),
    ARConfig(input_seq_len=256, num_examples=5_000, num_kv_pairs=64, num_queries=64),
]
test_configs = [
    ARConfig(input_seq_len=64, num_examples=1_000, num_kv_pairs=4, num_queries=4),
    ARConfig(input_seq_len=64, num_examples=1_000, num_kv_pairs=8, num_queries=8),
    ARConfig(input_seq_len=64, num_examples=1_000, num_kv_pairs=16, num_queries=16),
    ARConfig(input_seq_len=128, num_examples=1_000, num_kv_pairs=32, num_queries=32),
    ARConfig(input_seq_len=256, num_examples=1_000, num_kv_pairs=64, num_queries=64),
    ARConfig(input_seq_len=512, num_examples=1_000, num_kv_pairs=128, num_queries=128),
    ARConfig(input_seq_len=1024, num_examples=1_000, num_kv_pairs=256, num_queries=256),
]


synthetic_data = DataConfig
input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    num_train_examples=sum([c.num_examples for c in train_configs]),
    num_test_examples=sum([c.num_examples for c in test_configs]),
    vocab_size=VOCAB_SIZE,
    input_seq_len=input_seq_len,
    batch_size=(batch_size, batch_size / 8),
    force_cache=True,
    cache_dir="./var/sim_data/zg-synthetics",
    builder={
        "name": "zoology.data.ar_extrapolate.ar_extrapolate",
        "kwargs": {
            "train_configs": train_configs,
            "test_configs": test_configs
        }
    }   
)


def get_sequence_mixers(
    sequence_mixer: str,
    d_model: int
):
    # define this outside of if/else block because it is used in multiple mixers
    conv_mixer = dict(
        name="zoology.mixers.base_conv.BaseConv",
        kwargs={
            "l_max": input_seq_len,
            # pass a list of kernel sizes for each of four layers
            "kernel_size": 3,
            "implicit_long_conv": True,
        }
    )
    sequence_mixers = []
    if sequence_mixer == "lin-attn":
        block_type = "TransformerBlock"
        for ftr_map in [
            "none", 
            # "performer", 
            "cosformer", 
            # "taylor_exp"
        ]:
            for ftr_dim in [
                # 16, 32, 64, 128, 256, 
                # 512,
                1024
            ]:
                if ftr_map == "taylor_exp" and ftr_dim > 16:
                    continue

                lin_attn = dict(
                    name="zoology.mixers.based.Based",
                    kwargs={
                        "l_max": input_seq_len,
                        "feature_dim": ftr_dim,
                        "feature_name": ftr_map,
                        "num_key_value_heads": 1,
                        "num_heads": 1,
                        "train_view": "quadratic",
                    }
                )
                mixer = dict(
                    name="zoology.mixers.hybrid.Hybrid",
                    kwargs={"configs": [conv_mixer, lin_attn]}
                )
                name = f"lin-attn-{ftr_dim}-{ftr_map}"
                sequence_mixers.append((name, mixer))
    elif sequence_mixer == "attention-conv":
        block_type = "TransformerBlock"
        attention_mixer = dict(
            name="zoology.mixers.attention.MHA",
            kwargs={
                "dropout": 0.1,
                "num_heads": 1
            },
        )
        mixer = dict(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, attention_mixer]}
        )
        name = "attention-conv"
        sequence_mixers.append((name, mixer))
    else: 
        raise ValueError(f"Unknown sequence mixer {sequence_mixer}")

    return sequence_mixers, block_type


configs = []
for d_model in [
    # 64, 
    # 128, 
    256,
    512,
    # 1024,
    # 2048
]:
    for sequence_mixer in [
        "lin-attn",
        # "attention-conv",
        # "h3",
        # "mamba",
        # "hyena"
    ]:
        if sequence_mixer == "attention-conv" and d_model > 256:
            continue

        sequence_mixers, block_type = get_sequence_mixers(sequence_mixer, d_model)
        
        for name, mixer in sequence_mixers:
            # for lr in np.logspace(-4, -3, 4):
            for lr in np.logspace(-3, -1.5, 4):


                run_id = f"{name}-d_model={d_model}-lr={lr}"
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type=block_type,
                    max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=mixer,
                    state_mixer=dict(name="torch.nn.Identity", kwargs={})
                )
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=64,
                    logger=LoggerConfig(
                        project_name="zoology",
                        entity="hazy-research"
                    ),
                    sweep_id=sweep_name,
                    run_id=run_id,
                    predictions_path=f"./var/sim_data/zg-synthetics/predictions/{run_id}",
                    collect_predictions=True,
                )
                configs.append(config)