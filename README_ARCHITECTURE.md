# Arquitectura Completa del Pipeline

## Objetivo

Maximizar mejora de WER en ECU911 (sin pares clean/noisy) manteniendo
estabilidad acústica del generador y robustez del discriminador de WER.

## Stages

1.  `pretrain`
- Dataset principal: SPC (paired limpio/degradado por teléfono) cuando está
  disponible.
- Objetivo: que el generador aprenda enhancement sin degradar inteligibilidad.
- Loss total:
  - `L_mrstft`: Multi-Resolution STFT magnitude L1.
  - `L_si_sdr`: SI-SDR para fidelidad temporal.
  - `L_l1`: regularización waveform.
2.  `adversarial`
- Mantiene losses de reconstrucción y agrega GAN hinge.
- Discriminador Patch 1D (tiempo), generador U-Net 1D.
- Loss G:
  - `L_mrstft + 0.4\*L_si_sdr + 0.1\*L_adv`.
- Loss D:
  - hinge estándar sobre real/fake.
3.  `pretrain_discriminator`
- Entrena `D_WER` por separado sobre ECU911 (`purpose="wer_disc"`).
- Modelo: Whisper encoder congelado + chunking largo + attentive pooling + head
  MLP.
- Targets por defecto:
  - `q = sigmoid(-(log1p(min(wer, cap))-mu)/sigma)`.
  - `mu/sigma` calibrados en train split (mediana + IQR robusto).
- Loss por defecto:
  - `SmoothL1(sigmoid(logits), q)` \+ `rank_hard_pairs` intra-batch.
- Métrica de selección `best.pt`:
  - Spearman en val.
4.  `finetune`
- Entrena generador en ECU911 sin pares.
- `D_WER` se carga desde checkpoint y está congelado por defecto.
- Loss base (anti-degradación):
  - `L_stft_anchor = MRSTFT(enh, noisy)`
  - `L_identity = L1(enh, noisy)`
- Semantic anchor (opcional cada N steps):
  - distancia L1 en espacio encoder Whisper (modelo liviano de train).
- WER-adv (opcional):
  - ranking `D_WER(enh) > D_WER(noisy)` con warmup.

## Modelos

## Generator (`models/generator.py`)

- U-Net 1D con:
  - downsampling progresivo,
  - bottleneck residual blocks,
  - upsampling con skips,
  - salida tipo máscara residual sobre señal de entrada.
- Intención: denoise estable y evitar colapso (no inventar audio).

## GAN Discriminator (`models/discriminator.py`)

- Patch-style 1D conv.
- Produce score por batch (promedio temporal de patches).

## WER Discriminator (`models/wer_discriminator.py`)

- Encoder Whisper congelado (ligero para training).
- Procesa audios largos por chunks y agrega embeddings.
- Attentive pooling temporal + MLP para logit final.

## Evaluación WER (obligatoria)

Siempre con:

- `UDA-LIDI/openai-whisper-large-es_ecu911DM`

Implementado en:

- `trainers/common.py -> load_whisper_asr(..., for_evaluation=True)`
- `engine/config.py -> evaluation.whisper_model_name`

Validación usa transcripción long-form chunked con:

- normalización consistente,
- `condition_on_prev_tokens=False`.

## Multi-GPU (HPC 3x RTX A4000)

Asignación por defecto (`engine/devices.py`):

- `cuda:0`: entrenamiento G/D.
- `cuda:1`: D_WER + semantic anchor Whisper.
- `cuda:2`: Whisper de validación WER.

Ventajas:

- Evita OOM por cargar todo en una sola GPU.
- Permite evaluar WER mientras el entrenamiento usa otra GPU.

## Dataloader y datos

No se reescribe contrato de datos. Se mantiene:

- `ECU911Dataset`, `collate_ecu911`, `StratifiedWERBatchSampler`.
- `purpose="wer_disc"` para pretrain de D_WER.

Sanity check recomendado en HPC:

- `python scripts/hpc_data_sanity.py --config \<config.yaml> --check-spc`

## Checkpointing

Por stage:

- `last.pt`
- `best.pt`

Criterio `best`:

- `pretrain_discriminator`: mayor Spearman.
- resto: menor `wer_enh`.

## Curriculum recomendado

1.  `pretrain` en SPC hasta estabilizar pérdidas acústicas.
2.  `pretrain_discriminator` hasta Spearman val estable.
3.  `finetune` con:
- warmup inicial sin WER-adv,
- luego WER-adv cada N steps,
- D_WER congelado.
4.  opcional `adversarial` si la calidad perceptual cae.

## Ejecución

- `python train.py --stage pretrain`
- `python train.py --stage pretrain_discriminator`
- `python train.py --stage finetune --generator-ckpt ... --werd-ckpt ...`
- `python train.py --stage adversarial`

## Notas de riesgo

- Si el modelo UDA-LIDI no está cacheado en HPC, necesitas acceso HF o mirror
  local.
- Para audios extremadamente largos, reducir `evaluation.max_val_samples` para
  controlar tiempo.
- Si Spearman de D_WER se estanca:
  - subir batch de D_WER,
  - mantener sampler estratificado,
  - revisar `wer_filter_max_train`.
