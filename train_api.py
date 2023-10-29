import pathlib
import sys
import uuid

import asyncio
import websockets
from os import path
import json
import train_network_api
from loguru import logger

async def train_network(websocket, _path):
    trainer = train_network_api.NetworkTrainer()
    while True:
        raw_data = await websocket.recv()
        data = json.loads(raw_data)
        await websocket.send(
                json.dumps({
                    "status": 1001,
                    "message": "Lora scripts has received task request.",
                })
        )
        logger.info("Got task, starting train with train data: " + data["train_data"] + ", resolution: " + data["resolution"] + " model: " + data["model"])
        parser = train_network_api.setup_parser()
        args = parser.parse_args()
        output_name = str(uuid.uuid4())
        args.train_data_dir = path.join("train", data["train_data"])
        args.resolution = data["resolution"]
        if data["is_v2_model"]:
            args.v2 = True
        args.pretrained_model_name_or_path = path.join(
            "sd-models", data["model"] + ".safetensors"
        )
        args.output_name = output_name
        args = prepare_hardcoded_args(args, data["train_data"])
        await trainer.train(args, websocket, output_name)
        print("Finished a task...")
        break


def prepare_hardcoded_args(args, model):
    args.enable_bucket = True
    args.logging_dir = "./logs"
    args.log_prefix = model
    args.network_module = "networks.lora"
    args.max_train_epochs = 10
    args.learning_rate = 1e-4
    args.unet_lr = 1e-4
    args.text_encoder = 1e-5
    args.lr_scheduler = "cosine_with_restarts"
    args.lr_warmup_steps = 0
    args.lr_scheduler_num_cycles = 1
    args.network_dim = 128
    args.network_alpha = 128
    args.train_batch_size = 1
    args.save_every_n_epochs = 1
    args.mixed_precision = "fp16"
    args.save_precision = "fp16"
    args.seed = 1337
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
    args.output_dir = './output'
    return args


if __name__ == "__main__":
    start_server = websockets.serve(train_network, "localhost", 8998, ping_interval=10000, ping_timeout=10000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
