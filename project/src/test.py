from core import (
    parse_and_bootstrap,
    ArgSpec,
    create_and_load_model,
    logger,
    ModelConfig,
)

if __name__ == "__main__":
    args, cfg = parse_and_bootstrap(schema=[])
    model, checkpoint = create_and_load_model(
        model_config=ModelConfig(
            name="pcn", params={"num_dense": 16384, "latent_dim": 1024, "grid_size": 4}
        ),
        checkpoint_path=cfg.checkpoint_dir / "pcn_v69_best.pt",
        device=cfg.device,
    )
