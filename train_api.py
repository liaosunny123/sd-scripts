import pathlib
import sys
import uuid

from fastapi import FastAPI, WebSocket

import json
import train_network_api
from loguru import logger

app = FastAPI()


@app.get("/health")
async def health():
    return {
        "message": "OK!",
        "status_code": 200,
    }


@app.websocket("/train/network")
async def train_network(websocket: WebSocket):
    trainer = train_network_api.NetworkTrainer()
    await websocket.accept()
    while True:
        raw_data = await websocket.receive_text()
        data = json.loads(raw_data)
        await websocket.send_text(
            json.dumps(
                {
                    "status": 1001,
                    "message": "Lora scripts has received task request.",
                }
            )
        )
        logger.info("Got task, starting train with data" + data["train_data"])
        parser = train_network_api.setup_parser()
        output_name = str(uuid.uuid4())

        args_append = ["--train_data_dir", pathlib.Path.joinpath("train", data["train_data"]),
                       "--resolution", data["resolution"]]
        if data["is_v2_model"]:
            args_append.append("--v2")
        args_append.append(["--pretrained_model", pathlib.Path.joinpath(
            "../sd-models", data["model"] + ".safetensors"
        )])
        args_append.append(["--output_name", output_name])
        args_append = prepare_hardcoded_args(args_append, data["train_data"])
        args = parser.parse_args(args_append)
        await trainer.train(args, websocket)
        await websocket.send_text(
            json.dumps(
                {
                    "status": 3001,
                    "message": "Lora scripts has trained target object.",
                    "output": output_name,
                }
            )
        )


def prepare_hardcoded_args(args, model) -> list[str]:
    args.append([
        "--enable_bucket",
        "--logging_dir", ".logs",
        "--log_prefix", model,
        "--network_module", "networks.lora",
        "--max_train_epochs", "10",
        "--learning_rate", "1e-4",
        "--unet_lr", "1e-4r",
        "--text_encoder", "1e-5",
        "--lr_scheduler", "cosine_with_restarts",
        "--lr_warmup_steps", "0",
        "--lr_scheduler_num_cycles", "1",
        "--network_dim", "128",
        "--network_alpha", "128",
        "--train_batch_size", "1",
        "--save_every_n_epochs", "1",
        "--mixed_precision", "fp16",
        "--save_precision", "fp16",
        "--seed", "1337",
        "--cache_latents",
        "--prior_loss_weight", "1",
        "--max_token_length", "225",
        "--caption_extension", ".txt",
        "--save_model_as", "safetensors",
        "--min_bucket_reso", "256",
        "--max_bucket_reso", "1024",
        "--keep_tokens", "0",
        "--xformers",
        "--shuffle_caption",
        "--output_dir", '../output'
    ])
    return args


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8999)
