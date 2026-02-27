# mllm_service/tools.py

import typer
import uvicorn
from typing_extensions import Annotated
from .rr_process import insert_session
from .models_hub import (
    create_session,
    download_mllm_model,
    get_download_model_path,
)
from .network import MODEL_SESSION_CREATED

cli_app = typer.Typer()


@cli_app.command()
def main(
    host: Annotated[
        str, typer.Option(help="The host to bind the server to.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="The port to run the server on.")] = 8000,
    workers: Annotated[int, typer.Option(help="Number of worker processes.")] = 1,
    reload: Annotated[
        bool, typer.Option(help="Enable auto-reload for development.")
    ] = False,
    model_name: Annotated[str, typer.Option(help="Model Name")] = None,
):
    """
    Starts the MLLM Service using Uvicorn ASGI server.
    """
    print(f"🚀 Starting MLLM Service at http://{host}:{port}")

    if model_name is None:
        print("⚠️  Please specify a model name.")
        exit(1)

    # Insert name
    true_path = get_download_model_path(model_name)
    if true_path is not None:
        print(f"Found model at {true_path}")
        id, se = create_session(true_path)
        MODEL_SESSION_CREATED.add(id)
        insert_session(id, se)
    else:
        download_mllm_model(model_name)
        id, se = create_session(model_name)
        MODEL_SESSION_CREATED.add(id)
        insert_session(id, se)

    uvicorn.run(
        "pymllm.service.network:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


if __name__ == "__main__":
    cli_app()
