import argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lora_blip2.yaml')
    args = parser.parse_args()
    print('Training LoRA-BLIP2 with config:', args.config)
    # TODO: load YAML, build dataloader from data/style/past_captions.jsonl + images,
    # set up base BLIP-2 + LoRA adapters (PEFT), train, and save to models/checkpoints.
    # This file is a template to wire in your actual environment.

if __name__ == '__main__':
    main()
