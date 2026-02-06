# Isaac-GR00T-playground

GR00T N1.6 파인튜닝을 위한 프로젝트. NVIDIA Isaac-GR00T을 submodule로 관리하고, 커스텀 변경사항은 패치/오버레이로 분리.

## 프로젝트 구조

```
Isaac-GR00T-playground/
├── Isaac-GR00T/              # git submodule (e29d8fc) - 절대 수정 안함
├── custom/
│   ├── patches/              # upstream 수정사항 (docker build 시 git apply)
│   │   ├── 01-launch-finetune.patch          # torchvision 변환, gradient checkpointing, DeepSpeed ZeRO-3
│   │   ├── 02-experiment-single-gpu-deepspeed.patch  # 단일 GPU DeepSpeed 활성화
│   │   ├── 03-processing-override-keys.patch  # 이미지 변환 override_keys 추가
│   │   └── 04-zero3-offload-optimizer.patch   # optimizer CPU offload 설정
│   └── overlay/              # 새로 추가한 파일 (docker build 시 복사)
│       ├── examples/FFW_SG2/ffw_sg2_config.py
│       └── gr00t/configs/deepspeed/zero3_offload_config.json
├── docker/
│   ├── Dockerfile
│   └── build.sh
├── run_finetune.sh           # 학습 실행 스크립트
├── data/                     # 학습 데이터 (gitignored, 런타임 마운트)
└── outputs/                  # 학습 출력 (gitignored, 런타임 마운트)
```

## 새 데스크탑에서 시작하기

### 1. 클론

```bash
git clone --recurse-submodules https://github.com/kimtaehyeong99/Isaac-GR00T-playground.git
cd Isaac-GR00T-playground
```

> 이미 클론한 경우: `git submodule update --init`

### 2. 데이터 준비

`data/` 디렉토리에 LeRobot V2 형식 데이터셋 배치:

```
data/ffw_sg2_rev1_task_330_0205_jungmin_1/
├── meta/
│   ├── modality.json
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/chunk-000/
│   └── episode_*.parquet
└── videos/chunk-000/
    ├── observation.images.cam_head/
    ├── observation.images.cam_wrist_left/
    └── observation.images.cam_wrist_right/
```

### 3. Docker 이미지 빌드

```bash
bash docker/build.sh
```

- 이미지명: `gr00t-ffw`
- 빌드 컨텍스트: ~15MB (`.dockerignore`로 최적화)
- 소요 시간: ~5분 (pytorch3d 빌드 포함)

### 4. 학습 실행

```bash
docker run -d --name gr00t-finetune \
    --gpus all --shm-size=16g \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/run_finetune.sh:/workspace/run_finetune.sh \
    gr00t-ffw bash /workspace/run_finetune.sh
```

### 5. 모니터링

```bash
docker logs -f gr00t-finetune
```

## 학습 설정 요약

| 항목 | 값 |
|------|-----|
| 모델 | nvidia/GR00T-N1.6-3B |
| 배치 크기 | 4 (gradient accumulation: 32) |
| 학습률 | 1e-4 |
| 최대 스텝 | 5000 |
| DeepSpeed | ZeRO-3 + optimizer CPU offload |
| GPU 메모리 | ~6 GB (단일 GPU) |

## 하이퍼파라미터 변경

`run_finetune.sh`만 수정하면 됨 (이미지 리빌드 불필요, 런타임 마운트):

```bash
vi run_finetune.sh  # 파라미터 수정
docker rm -f gr00t-finetune  # 기존 컨테이너 삭제
# 다시 docker run ...
```

## Upstream 업데이트

```bash
cd Isaac-GR00T && git fetch origin && git checkout <new-tag>
# 패치 호환성 확인
for p in ../custom/patches/*.patch; do
    git apply --check "$p" && echo "OK: $p" || echo "NEEDS UPDATE: $p"
done
```
