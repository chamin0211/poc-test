"""
905명 홀터 파형 → 환자별 특징 9개 계산 (노트북 04a 셀 7, 8과 같은 방법)

실행:  python src/holter_features.py            # 905명 전체
       python src/holter_features.py --limit 3  # 3명만 시험
결과:  results/04/holter_features.csv  (환자 1명 끝날 때마다 한 줄씩 추가)
       results/04/holter_errors.csv    (처리에 실패한 환자와 오류 내용)
중간에 끊겨도 다시 실행하면 이미 끝난 환자는 건너뛰고 이어서 계산함
"""
import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk

warnings.filterwarnings("ignore")              # neurokit2 경고가 많아서 로그가 지저분해지지 않게

# ── 경로 ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # src/의 한 단계 위 = 프로젝트 폴더
DATA = ROOT / "data" / "music"
HOLTER = Path("/mnt/c/data/music/Holter_ECG")
OUT = ROOT / "results" / "04"
OUT_FEAT = OUT / "holter_features.csv"
OUT_ERR = OUT / "holter_errors.csv"


# ── 1) 대상 환자: 노트북 셀 1~3과 같은 기준 (960명 중 파형 있는 905명) ─────
def load_cohort():
    na = ["#VALUE!", "#value", "#N/A", "NA", ""]
    df = pd.read_csv(DATA / "subject-info.csv", sep=";", decimal=",", na_values=na)
    base = df.loc[~df["Exit of the study"].isin([1, 2])]                 # 추적탈락, 이식 제외
    base = base.loc[base["Age"].astype(str).str.strip() != ">89"]        # 나이 >89 제외
    ids = [pid for pid in base["Patient ID"] if (HOLTER / f"{pid}.hea").exists()]  # 파형 있는 사람만
    return ids


# ── 2) 셀 7과 같은 함수: 관례 규칙 + 가장 깨끗한 유도로 R파 검출 ─────────────
def nn_mask(rr):
    """정상 간격(NN)인지: 300~2000ms + 바로 앞 간격과 20% 이내 (Malik 필터)"""
    prev = np.r_[np.nan, rr[:-1]]
    return (rr >= 300) & (rr <= 2000) & (np.abs(rr - prev) <= 0.2 * prev)


def detect_r_full(pid):
    """모든 유도에서 R파를 검출하고 NN 비율이 가장 높은 유도의 결과를 돌려줌"""
    rec = wfdb.rdrecord(str(HOLTER / pid))
    fs = rec.fs
    n_nan = int(np.isnan(rec.p_signal).sum())  # 빈 값 개수 (지금까지 확인한 기록은 0)
    best, ratios = None, {}
    for i, lead in enumerate(rec.sig_name):    # 3유도면 X, Y, Z / 2유도면 X, Y
        clean = nk.ecg_clean(rec.p_signal[:, i], sampling_rate=fs)     # neurokit2 기본 필터
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)                # neurokit2 기본 R파 검출
        rpk = np.asarray(info["ECG_R_Peaks"])
        ratio = nn_mask(np.diff(rpk) / fs * 1000).mean() * 100
        ratios[lead] = ratio
        if best is None or ratio > best[0]:
            best = (ratio, lead, rpk)
    _, lead, rpk = best
    return rpk, fs, rec.sig_len, lead, ratios, n_nan


# ── 3) 셀 8과 같은 함수: R파 위치 → 특징 9개 ──────────────────────────────
def rr_features(rpk, fs):
    rr = np.diff(rpk) / fs * 1000
    t_end = rpk[1:] / fs
    nn = nn_mask(rr)
    x = rr[nn]
    pair = nn[:-1] & nn[1:]
    d = np.diff(rr)[pair]
    m5 = pd.Series(x).groupby((t_end[nn] // 300).astype(int)).mean()
    return {
        "minimum RR": x.min(),
        "Average RR": x.mean(),
        "maximum RR": x.max(),
        "RR range": x.max() - x.min(),
        "Longest RR pause": rr[rr >= 300].max(),
        "SDNN": x.std(),
        "SDANN": m5.std(),
        "RMSSD": np.sqrt(np.mean(d ** 2)),
        "pNN50": np.mean(np.abs(d) > 50) * 100,
        "NN 비율(%)": nn.mean() * 100,
    }


# ── 4) 환자 1명 처리: 다른 프로세스에서 실행되므로 결과를 dict로 돌려줌 ───────
def process(pid):
    t0 = time.time()
    try:
        rpk, fs, n, lead, ratios, n_nan = detect_r_full(pid)
        feat = rr_features(rpk, fs)
        row = {"Patient ID": pid, "유도": lead, "유도 수": len(ratios),
               "NN비율_X": ratios.get("X"), "NN비율_Y": ratios.get("Y"), "NN비율_Z": ratios.get("Z"),
               "기록 시간(h)": n / fs / 3600, "R파 수": len(rpk), "NaN 수": n_nan}
        row.update({f"wf_{k}": v for k, v in feat.items()})   # wf_ = waveform(파형)에서 계산한 값
        row["처리 시간(초)"] = time.time() - t0
        return row, None
    except Exception as e:
        return None, {"Patient ID": pid, "오류": repr(e)}


def append_row(path, row):
    """한 줄을 CSV 끝에 추가 (파일이 없으면 열 이름도 같이 씀)"""
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="앞에서 몇 명만 시험으로 돌릴지")
    ap.add_argument("--workers", type=int, default=4, help="동시에 처리할 환자 수 (1명당 메모리 약 1.5GB)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ids = load_cohort()
    done = set(pd.read_csv(OUT_FEAT)["Patient ID"]) if OUT_FEAT.exists() else set()   # 이미 끝난 환자
    todo = [p for p in ids if p not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"대상 {len(ids)}명 | 이미 완료 {len(done)}명 | 이번에 처리 {len(todo)}명 | 동시 {args.workers}명", flush=True)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:   # 환자 여러 명을 동시에 처리
        futures = {pool.submit(process, pid): pid for pid in todo}
        for k, fut in enumerate(as_completed(futures), 1):        # 끝나는 순서대로 받음
            row, err = fut.result()
            if row:
                append_row(OUT_FEAT, row)          # 1명 끝날 때마다 바로 저장 → 끊겨도 여기까지는 남음
                msg = f"유도 {row['유도']} | NN {row['wf_NN 비율(%)']:.1f}% | {row['처리 시간(초)']:.0f}초"
            else:
                append_row(OUT_ERR, err)
                msg = f"오류: {err['오류'][:80]}"
            elapsed = time.time() - t_start
            remain = elapsed / k * (len(todo) - k)  # 지금까지 속도로 남은 시간 추정
            print(f"[{k}/{len(todo)}] {futures[fut]} {msg} | 경과 {elapsed/60:.0f}분, 남은 시간 약 {remain/60:.0f}분", flush=True)

    print("완료:", OUT_FEAT, flush=True)


if __name__ == "__main__":
    main()
