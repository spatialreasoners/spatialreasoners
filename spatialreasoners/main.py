from .env import DEBUG

if DEBUG:
    import torch
    from jaxtyping import install_import_hook

    torch.set_printoptions(
        threshold=8,
        edgeitems=2
    )

    # Configure beartype and jaxtyping.
    with install_import_hook(
        ("src",),
        ("beartype", "beartype"),
    ):
        from spatialreasoners._main import main
else:
    from spatialreasoners._main import main


if __name__ == "__main__":
    main()
