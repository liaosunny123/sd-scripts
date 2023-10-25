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
        args = parser.parse_args()
        args.train_data_dir = pathlib.Path.joinpath("train", data["train_data"])
        args.resolution = data["resolution"]
        if data["is_v2_model"]:
            args.v2 = True
        args.pretrained_model = pathlib.Path.joinpath(
            "../sd-models", data["model"] + ".safetensors"
        )
        args.output_name = output_name
        args = prepare_hardcoded_args(args, data["train_data"])
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


def prepare_hardcoded_args(args, model):
    args.enable_bucket = True
    args.logging_dir = "./logs"
    args.log_prefix = model
    args.network_module = "networks.lora"
    args.max_train_epochs = 10
    args.learning_rate = "1e-4"
    args.unet_lr = "1e-4r"
    args.text_encoder = "1e-5"
    args.lr_scheduler = "cosine_with_restarts"
    args.lr_warmup_steps = 0
    args.lr_scheduler_num_cycles = 1
    args.network_dim = 128
    args.network_alpha = 128
    args.train_batch_size = 1
    args.save_every_n_epochs = 1
    args.mixed_precision = "fp16"
    args.save_precision = "fp16"
    args.seed = "1337"
    args.cache_latents = True
    args.prior_loss_weight = 1
    args.max_token_length = 225
    args.caption_extension = ".txt"
    args.save_model_as = "safetensors"
    args.min_bucket_reso = 256
    args.max_bucket_reso = 1024
    args.keep_tokens = 0
    args.xformers = True
    args.shuffle_caption = True
    args.output_dir = '../output'
    return args


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]))
