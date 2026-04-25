"""Legacy wrapper that reuses the live-env policy training entrypoint."""

try:
    from .train_grpo import main
except ImportError:
    from training.train_grpo import main


if __name__ == "__main__":
    main()
