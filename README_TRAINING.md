# Training Pipeline (Redesign)

## CLI (sin romper interfaz)

- `python train.py --stage pretrain`
- `python train.py --stage adversarial`
- `python train.py --stage pretrain_discriminator`
- `python train.py --stage finetune`

Opcionales:

- `\--config path/to/config.yaml`
- `\--resume checkpoints/.../last.pt`
- `\--generator-ckpt checkpoints/.../best.pt` (finetune)
- `\--werd-ckpt checkpoints/.../best.pt` (finetune)

## Checkpoints

Por stage se guardan:

- `checkpoints/\<experiment_name>/<stage>/last.pt`
- `checkpoints/\<experiment_name>/<stage>/best.pt`

Criterio de `best.pt`:

- `pretrain_discriminator`: mejor `spearman` en validación.
- `pretrain`, `adversarial`, `finetune`: menor `wer_enh`.

## Reanudar entrenamiento

- Reanudar desde `last.pt` automático:
  - `python train.py --stage finetune --resume
    checkpoints/\<exp>/finetune/last.pt`
- O cargar pesos previos al inicio de finetune:
  - `python train.py --stage finetune --generator-ckpt
    checkpoints/\<exp>/adversarial/best.pt --werd-ckpt
    checkpoints/\<exp>/pretrain_discriminator/best.pt`

## Replay offline (opcional)

Construcción de replay coherente con WER Whisper long-form:

- `python scripts/build_replay_buffer.py --generator-ckpt
  checkpoints/\<exp>/finetune/best.pt --out data/replay --max-samples 500`

Nota: finetune deja replay desactivado por default (`finetune.use_replay=false`).

## Verificación HPC de datos

- `python scripts/hpc_data_sanity.py --config path/to/config.yaml --check-spc`

Esto valida que:
- se encuentra `ecu911_prepared.pkl`,
- cargan splits train/val,
- el `purpose="wer_disc"` produce batches correctos,
- y (opcional) que SPC está accesible.

## Dónde van los datos

Con tu estructura actual, no necesitas mover nada. Usa `config.hpc.yaml` y deja:

- `data/processed/ecu911_prepared.pkl`
- `data/processed/clean_prepared.pkl`
- `data/processed/ecu911_16khz/...`
- `data/ecu911/audios/...`
- `data/ecu911/metadata.csv`
- `data/ecu911/test/noise_audios/...`
- `data/ecu911/test/metadata.csv`
- `data/spCSC/WAV/...`
- `data/spCSC/TXT/...`

Para entrenar:

- `python train.py --stage pretrain --config config.hpc.yaml`
- `python train.py --stage pretrain_discriminator --config config.hpc.yaml`
- `python train.py --stage finetune --config config.hpc.yaml --generator-ckpt <ckpt_G> --werd-ckpt <ckpt_DWER>`

## Diseño completo

Ver `README_ARCHITECTURE.md` para detalle de:
- modelos,
- losses,
- curriculum por stages,
- estrategia multi-GPU,
- criterios de selección de checkpoints.
