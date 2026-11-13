> Korean RunPod walkthrough written for the original working session (the "제가" voice is the coding agent). The English summary is in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md).

# 클라우드 GPU 대여 매뉴얼 (RunPod)

목적: AIMO 대회용 학습 데이터 생성(모델 5개로 문제 322개 풀이) + hidden state 추출 + 평가서버 속도 측정.
소요: 설정 15분, 실행 3~6시간. 예상 비용 $35~60.

## 왜 RunPod
- 평가 서버와 같은 GPU(**RTX PRO 6000 96GB, Blackwell**) 대여 가능 → 속도/메모리 측정이 그대로 통함.
- 볼륨이 `/workspace`에 마운트됨 → 제 스크립트가 이 경로 기준으로 작성됨.
- "SSH over exposed TCP" 기본 지원 → 제가 원격으로 전부 자동 실행.
- Vast.ai가 더 싸지만(개인 호스트 마켓) 디스크/네트워크/안정성 편차 큼. 마감 6일 전이라 안정성 우선.

## 1. 계정 + 충전 (5분)
1. https://www.runpod.io 가입 (Google 로그인 가능).
2. 왼쪽 메뉴 **Billing** → 카드로 **$60** 충전.
   - 선불 잔액이 곧 지출 상한. 잔액 0 되면 pod 자동 정지.

## 2. SSH 키 등록 (3분)
PowerShell에서:
```powershell
# 키 없을 때만 실행 (이미 %USERPROFILE%\.ssh\id_ed25519 있으면 생략). 암호 묻는 질문은 엔터 두 번.
ssh-keygen -t ed25519
# 공개키를 클립보드로 복사
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard
```
RunPod 콘솔 **Settings**(또는 **Credentials**) → **SSH Public Keys** 칸에 붙여넣고 저장.
- 붙여넣는 것은 `ssh-ed25519 AAAA...` 로 시작하는 **.pub(공개키)** 한 줄. 개인키(`.pub` 없는 파일)는 절대 어디에도 붙여넣지 말 것.

## 3. Pod 생성 (5분)
콘솔 **Pods** → **Deploy**.

| 항목 | 설정 |
|---|---|
| Cloud | **Secure Cloud** (Community는 더 싸지만 끊김/느린 다운로드 위험) |
| GPU | **RTX PRO 6000** (96GB). 목록에 "Server Edition"/"Workstation Edition" 둘 다 있으면 아무거나 |
| GPU Count | **4** (없으면 2, 그것도 없으면 아래 대안) |
| Template | **Runpod Pytorch** 최신 버전 (CUDA 12.8 이상 표기된 것) |
| Container Disk | **50 GB** |
| Volume Disk | **400 GB** (모델 5개 ≈130GB + 환경 + 결과) |
| Expose TCP Ports | `22` 포함 확인 (템플릿 기본값) |
| Additional filters → CUDA Versions | 12.8 이상만 체크 (가능하면 13.x 포함) |

GPU/디스크 설정은 템플릿 선택 후 **Edit Template**(또는 Pod 설정 화면)에서 변경 → **Deploy On-Demand**.

**재고 없을 때 대안 (위에서부터)**
1. RTX PRO 6000 × 2
2. H100 80GB (SXM 또는 PCIe) × 2~4 — 생성 속도는 오히려 빠름. 평가 HW 측정만 나중에 짧게 따로.
3. RTX PRO 6000 × 1 — 가능은 하지만 8시간+ 소요.

## 4. 접속 정보 전달 (1분)
Pod 목록에서 생성한 Pod 클릭 → **Connect** 탭 → **SSH over exposed TCP** 아래 명령 복사:
```
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```
이 한 줄을 채팅에 붙여넣기. (IP/포트만 담긴 정보라 공유해도 키 유출 없음.)
- **Basic SSH**(`ssh <pod-id>@ssh.runpod.io ...`) 말고 **exposed TCP** 쪽 명령이어야 함.

## 5. 이후 (자동, 제가 진행)
1. 접속 확인 → 코드/데이터 업로드 → 환경 설치(torch 2.12.1 + transformers 5.13.0 런타임 복제, vLLM).
2. 모델 5개 다운로드 → GPU별로 풀이 생성 병렬 실행 → hidden state 추출 → 속도 벤치.
3. 결과 로컬로 회수 → probe 학습 → 제출 ZIP.
4. 끝나면 보고드림. 진행 상황은 수시로 공유.

## 6. 비용 관리 / 종료
- 과금은 **Pod가 Running인 동안** GPU 단가 × 장수. 4장 기준 시간당 약 $8~9.
- 작업 완료 보고 받으면 콘솔에서 **Stop**(□) → 결과 회수 확인 후 **Terminate**(휴지통).
  - Stop: GPU 과금 멈춤, 볼륨(`/workspace`) 보관료만 소액 과금. 재시작 시 같은 GPU가 없으면 GPU 0장으로 뜰 수 있음.
  - Terminate: 볼륨까지 삭제, 과금 완전 종료. **제가 "회수 완료" 말하기 전에는 Terminate 금지.**
- 이상 과금 의심 시: 콘솔 Pods에서 바로 Stop.

## 문제 해결
- 접속 시 비밀번호를 물음 → 공개키 미등록/오등록. 2단계 다시 확인 후 Pod **재시작**(키는 Pod 시작 시 주입됨).
- "OCI runtime create failed" → Additional filters → CUDA Versions에서 12.8+ 다시 선택 후 재생성.
- RTX PRO 6000이 목록에 없음 → 3단계 대안 순서대로.
