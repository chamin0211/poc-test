# poc 환경 사용법 (WSL + venv + H2O)

## 매번 작업 시작할 때

```bash
cd ~/poc
source .venv/bin/activate     # 프롬프트에 (.venv)가 뜨면 성공
./start_h2o.sh                # 출력에 "cloud_size":1 이 나와야 정상
```

## 작업 끝낼 때

```bash
pkill -f h2o.jar              # 안 끄면 메모리 4GB를 계속 잡는다
pgrep -fa h2o.jar             # 아무것도 안 나오면 종료된 것
```

## 코드에서 H2O 쓰는 법

```python
import h2o

# init()이 아니라 connect()를 쓴다. init()은 서버를 새로 띄우려 해서 멈춘다.
h2o.connect(ip="127.0.0.1", port=54321)
```

## 왜 start_h2o.sh가 필요한가

- h2o.init()으로 서버를 띄우면 존재하지 않는 .254:54321 노드가 클러스터 멤버로 잡혀서 cloud_size가 2가 되고, 연산 단계에서 멈췄다.
- 멤버 목록을 자기 자신 하나로 고정하는 -flatfile 옵션을 주면 cloud_size가 1이 된다. (근본 원인은 확인 필요)
- WSL을 재시작하면 서버가 사라지므로 ./start_h2o.sh로 다시 띄운다.

## 폴더 구조

```
~/poc
  .venv/            가상환경 (Git에 올리지 않음)
  data/music/       subject-info*.csv, RECORDS (Git에 올리지 않음)
  src/              스크립트
  notebooks/        노트북
  start_h2o.sh      H2O 서버 시작 스크립트
  requirements.txt  패키지 목록
```

## 데이터 가져오기 (Windows 다운로드 폴더에서 복사)

```bash
mkdir -p ~/poc/data/music
cp /mnt/c/Users/Admin/Downloads/subject-info.csv \
   /mnt/c/Users/Admin/Downloads/subject-info_codes.csv \
   /mnt/c/Users/Admin/Downloads/subject-info_definitions.csv \
   /mnt/c/Users/Admin/Downloads/RECORDS \
   ~/poc/data/music/
ls -lh ~/poc/data/music
```

- 원본을 엑셀로 열어서 저장하지 않는다. 값 형식이 바뀌거나 #VALUE! 같은 문자열이 생길 수 있다.
- 데이터 파일은 재배포 금지라서 data/ 는 .gitignore에 둔다.

## 문제 생겼을 때

| 증상 | 확인 |
|---|---|
| connect 후 멈춤 | curl -s http://127.0.0.1:54321/3/Cloud 로 cloud_size가 1인지 확인. 2면 ./start_h2o.sh로 재시작 |
| 포트 사용 중 | pkill -f h2o.jar 후 다시 시작 |
| 프롬프트에 (base)가 뜸 | conda deactivate |
| ModuleNotFoundError | (.venv)가 켜져 있는지 확인 후 pip install -r requirements.txt |
