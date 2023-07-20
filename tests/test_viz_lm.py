import os
import tempfile

import jax
import pytest
from ray.util.client import ray

import haliax

import levanter.main.viz_logprobs as viz_logprobs
import tiny_test_corpus
from levanter.checkpoint import save_checkpoint
from levanter.distributed import RayConfig
from levanter.logging import WandbConfig
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


def setup_module(module):
    ray.init("local", num_cpus=10)


def teardown_module(module):
    ray.shutdown()


@pytest.mark.entry
def test_viz_lm():
    # just testing if eval_lm has a pulse
    # save a checkpoint
    model_config = Gpt2Config(
        num_layers=2,
        num_heads=2,
        seq_len=128,
        hidden_dim=32,
    )

    with tempfile.TemporaryDirectory() as f:
        try:
            data_config = tiny_test_corpus.tiny_corpus_config(f)
            tok = data_config.the_tokenizer
            Vocab = haliax.Axis("vocab", len(tok))
            model = Gpt2LMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))

            save_checkpoint(model, None, 0, f"{f}/ckpt")

            config = viz_logprobs.VizGpt2Config(
                data=data_config,
                model=model_config,
                trainer=viz_logprobs.TrainerConfig(
                    per_device_eval_parallelism=len(jax.devices()),
                    max_eval_batches=1,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                ),
                checkpoint_path=f"{f}/ckpt",
                num_docs=len(jax.devices()),
                output_dir=f"{f}/viz",
            )
            viz_logprobs.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass