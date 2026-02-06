# GR00T N1.6 파인튜닝 작업 로그

> 모든 작업을 시간순으로 기록합니다.

---

## 2026-02-06

### 09:XX - 계획 수립 완료
**수행 내용:**
- 레파지토리 구조 탐색 (Isaac-GR00T/)
- Docker 지원 확인 (docker/Dockerfile, docker/build.sh)
- 학습 스크립트 분석 (gr00t/experiment/launch_finetune.py)
- 사용자 데이터셋 분석 (ffw_sg2_rev1_task_330_0205_jungmin_1)

**분석 결과:**
- 에피소드: 5개, 6632 프레임, 15 FPS
- 카메라: cam_head, cam_wrist_left, cam_wrist_right (3개)
- 상태/액션 차원: 22

**계획 파일:** `/home/robotis-ai/.claude/plans/nested-crafting-mochi.md`

---

### 09:XX - 1단계 완료: modality.json 생성 ✅
**목적:** 데이터셋에 GR00T 학습에 필요한 메타데이터 파일 추가
**파일 경로:** `/home/robotis-ai/gr00t_ws/data/ffw_sg2_rev1_task_330_0205_jungmin_1/meta/modality.json`

**생성된 구조:**
- state: arm_left(0-7), gripper_left(7-8), arm_right(8-15), gripper_right(15-16), head(16-18), lift(18-19), mobile_base(19-22)
- action: state와 동일
- video: cam_head, cam_wrist_left, cam_wrist_right
- annotation: human.task_description

---

### 09:XX - 2단계 완료: ffw_sg2_config.py 생성 ✅
**목적:** 커스텀 로봇의 modality config 정의
**파일 경로:** `/home/robotis-ai/gr00t_ws/Isaac-GR00T/examples/FFW_SG2/ffw_sg2_config.py`

**설정 내용:**
- video: cam_head, cam_wrist_left, cam_wrist_right
- state: arm_left, gripper_left, arm_right, gripper_right, head, lift, mobile_base
- action: 16-step horizon, 각 modality별 ActionConfig 정의
  - 관절(arm, head, lift): RELATIVE 표현
  - 그리퍼, 모바일베이스: ABSOLUTE 표현
- language: annotation.human.task_description

---

### 09:XX - 3단계 완료: Docker 이미지 빌드 ✅
**목적:** GR00T 학습 환경 Docker 이미지 빌드
**명령어:** `cd Isaac-GR00T/docker && bash build.sh`
**참고:** 사용자가 기존 이미지 삭제 후 새로 빌드 요청

**결과:**
- 이미지명: `gr00t-dev:latest`
- 이미지 크기: 41.8GB
- 베이스 이미지: `nvcr.io/nvidia/pytorch:25.04-py3`
- 상태: **빌드 성공** (Image gr00t-dev BUILT SUCCESSFULLY)

---

### XX:XX - 4단계: Docker 컨테이너 실행 및 파인튜닝
**목적:** 컨테이너 실행 후 GR00T N1.6 파인튜닝 시작
**실행 방식:** 사용자 직접 실행

#### Step 1: Docker 컨테이너 실행
```bash
docker run -it --rm \
    --gpus all \
    --shm-size=16g \
    -v /home/robotis-ai/gr00t_ws/Isaac-GR00T:/workspace/gr00t \
    -v /home/robotis-ai/gr00t_ws/data:/workspace/data \
    -v /home/robotis-ai/gr00t_ws/outputs:/workspace/outputs \
    gr00t-dev /bin/bash
```

#### Step 2: 패키지 설치 (최초 1회)
```bash
cd /workspace/gr00t
pip install tyro --break-system-packages
```

#### Step 3: 파인튜닝 실행
```bash
cd /workspace/gr00t

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /workspace/data/ffw_sg2_rev1_task_330_0205_jungmin_1 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /workspace/gr00t/examples/FFW_SG2/ffw_sg2_config.py \
    --num-gpus 1 \
    --output-dir /workspace/outputs/ffw_sg2_finetune \
    --global-batch-size 32 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --max-steps 5000 \
    --save-steps 500 \
    --save-total-limit 5 \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --weight-decay 1e-5 \
    --warmup-ratio 0.05
```

#### CUDA OOM 발생 시 (메모리 부족):
```bash
--global-batch-size 16 --gradient-accumulation-steps 8
```

---

### XX:XX - 4단계 트러블슈팅: 패키지 의존성 문제 해결
**문제:** Docker 볼륨 마운트로 인해 이미지 내 설치된 패키지가 무효화됨
**원인:** `-v` 마운트가 이미지 내부의 `/workspace/gr00t/` 디렉토리를 덮어씀

**해결 방법:** 컨테이너 내에서 누락된 패키지 일괄 설치

```bash
# 필요한 패키지 설치
pip install --break-system-packages --ignore-installed --no-deps \
    tyro==0.9.17 \
    omegaconf==2.3.0 \
    lmdb==1.7.5 \
    msgpack-numpy==0.4.8 \
    termcolor==3.2.0 \
    gymnasium==1.2.2 \
    av==15.0.0

pip install --break-system-packages docstring-parser shtab "antlr4-python3-runtime==4.9.3"
pip install --break-system-packages "transformers==4.51.3"

# PYTHONPATH 설정
export PYTHONPATH=/workspace/gr00t:$PYTHONPATH
```

---

### 01:00 - 4단계 완료: 파인튜닝 실행 중 ✅
**상태:** 백그라운드에서 학습 진행 중
**컨테이너명:** gr00t-finetune
**로그 파일:** `/home/robotis-ai/gr00t_ws/outputs/finetune.log`

**모니터링 명령어:**
```bash
# 실시간 로그 확인
tail -f /home/robotis-ai/gr00t_ws/outputs/finetune.log

# 컨테이너 상태 확인
docker ps | grep gr00t-finetune

# Docker 로그 확인
docker logs -f gr00t-finetune
```

**학습 설정:**
- 모델: nvidia/GR00T-N1.6-3B
- 배치 크기: 32 (gradient accumulation: 4)
- 학습률: 1e-4
- 최대 스텝: 5000
- 체크포인트 저장: 500 스텝마다

**출력 디렉토리:** `/home/robotis-ai/gr00t_ws/outputs/ffw_sg2_finetune/`

---

### XX:XX - 4단계 오류: 이미지 크기 불일치
**오류 메시지:**
```
RuntimeError: stack expects each tensor to be equal size, but got [1, 3, 256, 458] at entry 0 and [1, 3, 256, 456] at entry 1
```

**원인 분석:**
- 카메라별 종횡비 차이:
  - cam_head: 376x672 (비율 ~1.79)
  - cam_wrist: 240x424 (비율 ~1.77)
- albumentations의 `SmallestMaxSize`가 종횡비를 유지하면서 리사이즈하여 서로 다른 크기 출력
- 이미지를 stack할 때 크기가 달라서 오류 발생

**해결 방법:**
- `use_albumentations_transforms=False`로 설정하여 torchvision 변환 사용
- torchvision 변환은 `LetterBoxTransform`으로 먼저 정사각형으로 패딩 후 리사이즈하여 크기 통일
- `shortest_image_edge=None`, `crop_fraction=None` 설정 필요
- 대신 `image_crop_size`, `image_target_size` 설정

---

### XX:XX - launch_finetune.py 수정
**목적:** 이미지 크기 불일치 해결을 위해 torchvision 변환 사용
**파일:** `/home/robotis-ai/gr00t_ws/Isaac-GR00T/gr00t/experiment/launch_finetune.py`

**수정 내용:**
- `config.model.use_albumentations_transforms = False` 추가
- `config.model.shortest_image_edge = None` 추가
- `config.model.crop_fraction = None` 추가
- `config.model.image_crop_size = (224, 224)` 추가
- `config.model.image_target_size = (256, 256)` 추가

---

### XX:XX - processing_gr00t_n1d6.py 수정 (추가)
**문제:** `from_pretrained`에서 `use_albumentations` 등의 설정이 오버라이드되지 않음
**파일:** `/home/robotis-ai/gr00t_ws/Isaac-GR00T/gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py`

**수정 위치:** 513-516행, `override_keys` 리스트

**수정 전:**
```python
override_keys = [
    "random_rotation_angle",
    "color_jitter_params",
    "use_relative_action",
]
```

**수정 후:**
```python
override_keys = [
    "random_rotation_angle",
    "color_jitter_params",
    "use_relative_action",
    "use_albumentations",
    "image_crop_size",
    "image_target_size",
    "shortest_image_edge",
    "crop_fraction",
]
```

---

### XX:XX - CUDA 메모리 부족 해결
**문제:** 학습 시작 시 CUDA out of memory 오류 발생
**원인:** GPU 메모리 31.35 GiB 중 30.11 GiB 사용 (거의 최대)
**파일:** `/home/robotis-ai/gr00t_ws/run_finetune.sh`

**수정 내용:**
- `--global-batch-size 32` → `--global-batch-size 8`
- `--gradient-accumulation-steps 4` → `--gradient-accumulation-steps 16`
- 실제 배치 크기: 8/16 = 0.5 (per GPU), 누적 효과로 effective batch size 8 유지

---

### XX:XX - CUDA 메모리 추가 최적화
**문제:** batch size 8로도 여전히 OOM 발생
**해결:** 추가 메모리 최적화 적용

**수정 내용:**
- `--global-batch-size 8` → `--global-batch-size 4`
- `--gradient-accumulation-steps 16` → `--gradient-accumulation-steps 32`
- `--dataloader-num-workers 4` → `--dataloader-num-workers 2`
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 추가

---

### XX:XX - Gradient Checkpointing 활성화
**문제:** 모델이 GPU 메모리 29GB+ 사용, 배치 크기 줄여도 OOM 발생
**해결:** Gradient checkpointing 활성화하여 메모리-연산 트레이드오프

**파일:** `/home/robotis-ai/gr00t_ws/Isaac-GR00T/gr00t/experiment/launch_finetune.py`

**수정 내용:**
```python
config.training.gradient_checkpointing = True  # Enable gradient checkpointing to save GPU memory
```

---

### XX:XX - DeepSpeed ZeRO-3 + Optimizer Offload 활성화
**문제:** Gradient checkpointing만으로 OOM 해결 안됨 (optimizer states가 GPU 메모리 소모)
**해결:** DeepSpeed ZeRO-3 + optimizer offload to CPU

**수정 파일 1:** `/home/robotis-ai/gr00t_ws/Isaac-GR00T/gr00t/configs/deepspeed/zero3_config.json`
- `offload_optimizer` 설정 추가 (CPU로 optimizer states 오프로드)

**수정 파일 2:** `/home/robotis-ai/gr00t_ws/Isaac-GR00T/gr00t/experiment/launch_finetune.py`
- `config.training.deepspeed_stage = 3` 추가

**수정 파일 3:** `/home/robotis-ai/gr00t_ws/Isaac-GR00T/gr00t/experiment/experiment.py`
- 단일 GPU에서도 DeepSpeed 활성화 (기존에는 num_gpus > 1 조건 필요)
- 기존: `if config.training.num_gpus > 1 and not config.training.use_ddp:`
- 변경: `if not config.training.use_ddp:`

**결과:** GPU 메모리 사용량 ~6.12 GB로 감소 (기존 29+ GB), 학습 정상 시작

---

### XX:XX - 프로젝트 구조 개편: Submodule + Patch 방식

**목적:** Isaac-GR00T을 submodule로 전환, upstream 코드는 절대 수정하지 않고, Docker 빌드 시 패치/오버레이 적용

**작업 내용:**

1. **기존 Isaac-GR00T 클론 제거** → git submodule로 교체 (커밋 `e29d8fc`)
2. **패치 파일 4개 생성** (upstream 파일 수정사항):
   - `custom/patches/01-launch-finetune.patch` — torchvision 변환, gradient checkpointing, DeepSpeed ZeRO-3
   - `custom/patches/02-experiment-single-gpu-deepspeed.patch` — 단일 GPU DeepSpeed 활성화
   - `custom/patches/03-processing-override-keys.patch` — 이미지 변환 override_keys 추가
   - `custom/patches/04-zero3-offload-optimizer.patch` — optimizer CPU offload 설정
3. **오버레이 파일 2개 배치** (새로 생성한 파일):
   - `custom/overlay/examples/FFW_SG2/ffw_sg2_config.py` — 커스텀 로봇 modality config
   - `custom/overlay/gr00t/configs/deepspeed/zero3_offload_config.json` — 대체 DeepSpeed 설정
4. **Docker 파일 생성**:
   - `docker/Dockerfile` — 커스텀 빌드 (의존성 베이킹 + 패치 적용)
   - `docker/build.sh` — 빌드 스크립트
5. **run_finetune.sh 정리** — pip install 제거 (이미지에 베이킹됨)
6. **`.gitignore`, `.dockerignore` 생성**

**최종 프로젝트 구조:**
```
Isaac-GR00T-playground/
├── Isaac-GR00T/          (git submodule - 절대 수정 안함)
├── custom/
│   ├── patches/          (upstream 수정사항 4개)
│   └── overlay/          (새 파일 2개)
├── docker/
│   ├── Dockerfile
│   └── build.sh
├── run_finetune.sh       (학습 실행 - pip install 제거됨)
├── WORK_LOG.md
├── .gitignore
├── .dockerignore
├── .gitmodules
├── data/                 (gitignored, 런타임 마운트)
└── outputs/              (gitignored, 런타임 마운트)
```

**Docker 이미지:** `gr00t-ffw` (42GB) — 빌드 성공 ✅

**Docker 실행 방식:**
```bash
docker run -d --name gr00t-finetune \
    --gpus all --shm-size=16g \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/run_finetune.sh:/workspace/run_finetune.sh \
    gr00t-ffw bash /workspace/run_finetune.sh
```

**Upstream 업데이트 시:**
```bash
cd Isaac-GR00T && git fetch origin && git checkout <new-tag>
for p in ../custom/patches/*.patch; do
    git apply --check "$p" && echo "OK: $p" || echo "NEEDS UPDATE: $p"
done
```
