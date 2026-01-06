#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lotofácil — Gerador de apostas (15 dezenas) com:
- Histórico local com atualização incremental (cache)
- Tabela de percentuais por dezena (01–25)
- Tabela de par/ímpar (global + distribuição de pares por concurso)
- Geração de N apostas ponderadas pelos percentuais e pelo padrão par/ímpar
- Regra crítica: NÃO pode existir nenhum concurso histórico com 14 ou 15 dezenas em comum com a aposta

Uso:
  python lotofacil_gerador.py --n 5
  python lotofacil_gerador.py --n 5 --seed 123
  python lotofacil_gerador.py --n 10 --cache-dir ./dados --saida apostas.txt

Observação:
  - O programa tenta atualizar o histórico via endpoints públicos.
  - Se não houver internet, ele usa o cache local (se existir).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


BASE_URL_ULTIMO = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
BASE_URL_CONCURSO = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil/{numero}"


# ----------------------------
# Utilitários HTTP / Cache
# ----------------------------

def http_get_json(url: str, timeout: int = 15, retries: int = 5, backoff_base: float = 0.8) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; lotofacil_gerador/1.0)",
        "Accept": "application/json,text/plain,*/*",
    }
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < retries:
                sleep_s = backoff_base * (2 ** (attempt - 1)) + random.random() * 0.2
                time.sleep(sleep_s)
            else:
                break
    raise RuntimeError(f"Falha ao buscar JSON: {url} | erro: {last_err!r}")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cache(cache_file: Path) -> dict:
    if not cache_file.exists():
        return {"concursos": [], "atualizado_em": None}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if "concursos" not in data or not isinstance(data["concursos"], list):
            raise ValueError("Formato inválido do cache (campo 'concursos').")
        return data
    except Exception as e:
        raise RuntimeError(f"Falha ao ler cache {cache_file}: {e}")


def save_cache(cache_file: Path, concursos: List[dict]) -> None:
    data = {"concursos": concursos, "atualizado_em": now_iso()}
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_dezenas_from_api(payload: dict) -> List[int]:
    """
    O endpoint normalmente retorna lista em 'listaDezenas' (strings) para concursos.
    Ex.: ["01","02",...]
    """
    dezenas = payload.get("listaDezenas")
    if not dezenas or not isinstance(dezenas, list):
        raise ValueError("Resposta sem 'listaDezenas' ou formato inesperado.")
    out = []
    for x in dezenas:
        if isinstance(x, str):
            out.append(int(x))
        elif isinstance(x, int):
            out.append(x)
        else:
            raise ValueError("Dezena em formato inesperado.")
    if len(out) != 15:
        raise ValueError(f"Esperado 15 dezenas, veio {len(out)}.")
    # valida faixa
    for d in out:
        if d < 1 or d > 25:
            raise ValueError(f"Dezena fora do intervalo 1..25: {d}")
    return out


def get_numero_concurso(payload: dict) -> int:
    n = payload.get("numero")
    if isinstance(n, int):
        return n
    if isinstance(n, str) and n.isdigit():
        return int(n)
    raise ValueError("Resposta sem 'numero' do concurso.")


def carregar_historico(cache_dir: Path, atualizar: bool = True) -> List[dict]:
    """
    Retorna lista de concursos no formato:
      {"concurso": int, "dezenas": [int,...15]}
    Atualiza incrementalmente se cache existir.
    """
    ensure_dir(cache_dir)
    cache_file = cache_dir / "lotofacil_historico.json"

    cache = load_cache(cache_file)
    concursos: List[dict] = cache.get("concursos", [])
    # normalizar (pode ter vindo vazio)
    if not isinstance(concursos, list):
        concursos = []

    # remover duplicatas e ordenar, caso necessário
    tmp_map: Dict[int, List[int]] = {}
    for c in concursos:
        try:
            num = int(c["concurso"])
            dezenas = list(map(int, c["dezenas"]))
            if len(dezenas) == 15:
                tmp_map[num] = dezenas
        except Exception:
            continue
    concursos = [{"concurso": k, "dezenas": tmp_map[k]} for k in sorted(tmp_map.keys())]

    if not atualizar:
        if not concursos:
            raise RuntimeError(
                "Cache não existe ou está vazio e a atualização foi desativada. "
                "Execute sem '--no-update' para baixar o histórico."
            )
        return concursos

    # descobrir último salvo
    ultimo_salvo = max((c["concurso"] for c in concursos), default=0)

    # descobrir último concurso online
    try:
        payload_ultimo = http_get_json(BASE_URL_ULTIMO)
        ultimo_online = get_numero_concurso(payload_ultimo)
    except Exception as e:
        if concursos:
            # Sem internet, segue com cache
            print(f"[AVISO] Não foi possível atualizar histórico. Usando cache local. Detalhe: {e}", file=sys.stderr)
            return concursos
        raise RuntimeError(
            "Não foi possível obter histórico (sem internet e sem cache local). "
            "Conecte à internet e execute novamente."
        )

    if ultimo_salvo >= ultimo_online:
        # nada a fazer
        if not concursos and ultimo_salvo == 0:
            # Edge (pouco provável): endpoint ok mas cache vazio -> baixar tudo
            pass
        else:
            return concursos

    # Se não houver cache (ultimo_salvo==0), baixar do 1 até o último
    inicio = 1 if ultimo_salvo == 0 else (ultimo_salvo + 1)
    fim = ultimo_online

    print(f"[INFO] Atualizando histórico: baixando concursos {inicio}..{fim}", file=sys.stderr)

    novos: Dict[int, List[int]] = {}
    # Se inicio==fim, podemos usar o payload_ultimo quando for o mesmo
    for num in range(inicio, fim + 1):
        try:
            if num == ultimo_online:
                payload = payload_ultimo
            else:
                payload = http_get_json(BASE_URL_CONCURSO.format(numero=num))
            numero = get_numero_concurso(payload)
            dezenas = parse_dezenas_from_api(payload)
            novos[numero] = dezenas
            print(f"[INFO] Baixado concurso {numero}", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(f"Falha ao baixar concurso {num}: {e}")

    # mesclar
    for c in concursos:
        novos.setdefault(int(c["concurso"]), list(map(int, c["dezenas"])))

    merged = [{"concurso": k, "dezenas": novos[k]} for k in sorted(novos.keys())]
    save_cache(cache_file, merged)
    return merged


# ----------------------------
# Estatísticas / Tabelas
# ----------------------------

@dataclass(frozen=True)
class Stats:
    freq_abs: Dict[int, int]              # contagem por dezena
    freq_pct: Dict[int, float]            # percentual por dezena
    pares_pct: float
    impares_pct: float
    dist_pares_por_concurso: Dict[int, int]  # quantos concursos tiveram X pares


def calcular_stats(historico: List[dict]) -> Stats:
    if not historico:
        raise ValueError("Histórico vazio.")

    freq_abs = {d: 0 for d in range(1, 26)}
    total_dezenas = len(historico) * 15
    total_pares = 0
    total_impares = 0
    dist_pares: Dict[int, int] = {}

    for c in historico:
        dezenas = list(map(int, c["dezenas"]))
        pares_no_concurso = 0
        for d in dezenas:
            freq_abs[d] += 1
            if d % 2 == 0:
                total_pares += 1
                pares_no_concurso += 1
            else:
                total_impares += 1
        dist_pares[pares_no_concurso] = dist_pares.get(pares_no_concurso, 0) + 1

    freq_pct = {d: (freq_abs[d] / total_dezenas) * 100.0 for d in range(1, 26)}
    pares_pct = (total_pares / total_dezenas) * 100.0
    impares_pct = (total_impares / total_dezenas) * 100.0

    return Stats(
        freq_abs=freq_abs,
        freq_pct=freq_pct,
        pares_pct=pares_pct,
        impares_pct=impares_pct,
        dist_pares_por_concurso=dict(sorted(dist_pares.items(), key=lambda x: (-x[1], x[0]))),
    )


def imprimir_tabela_dezenas(stats: Stats) -> str:
    rows = sorted(stats.freq_pct.items(), key=lambda x: (-x[1], x[0]))
    out = []
    out.append("Tabela — Percentual por dezena (01–25) considerando todos os concursos")
    out.append("-" * 72)
    out.append(f"{'Dezena':>6} | {'Qtde':>6} | {'Percentual':>10}")
    out.append("-" * 72)
    for d, pct in rows:
        out.append(f"{d:02d}     | {stats.freq_abs[d]:6d} | {pct:9.4f}%")
    out.append("-" * 72)
    return "\n".join(out)


def imprimir_tabela_par_impar(stats: Stats) -> str:
    out = []
    out.append("Tabela — Par/Ímpar (global) considerando todos os concursos")
    out.append("-" * 72)
    out.append(f"Pares:   {stats.pares_pct:.4f}%")
    out.append(f"Ímpares: {stats.impares_pct:.4f}%")
    out.append("-" * 72)
    out.append("Distribuição — Quantidade de PARES por concurso (recomendado p/ geração)")
    out.append("-" * 72)
    out.append(f"{'Pares no concurso':>16} | {'Concursos':>8}")
    out.append("-" * 72)
    for k, v in stats.dist_pares_por_concurso.items():
        out.append(f"{k:16d} | {v:8d}")
    out.append("-" * 72)
    return "\n".join(out)


# ----------------------------
# Geração e validação
# ----------------------------

def dezenas_para_mask(dezenas: List[int]) -> int:
    mask = 0
    for d in dezenas:
        mask |= 1 << (d - 1)  # bit 0 => dezena 1
    return mask


def preparar_masks_historico(historico: List[dict]) -> List[int]:
    return [dezenas_para_mask(list(map(int, c["dezenas"]))) for c in historico]


def sample_k_from_distribution(dist: Dict[int, int], rng: random.Random) -> int:
    """
    dist: {k_pares: quantidade_de_concursos}
    retorna um k_pares amostrado proporcionalmente.
    """
    if not dist:
        # fallback razoável
        return rng.choice([7, 8])
    items = sorted(dist.items(), key=lambda x: x[0])
    ks = [k for k, _ in items]
    ws = [w for _, w in items]
    total = sum(ws)
    r = rng.uniform(0, total)
    acc = 0.0
    for k, w in zip(ks, ws):
        acc += w
        if r <= acc:
            return k
    return ks[-1]


def normalize_weights(nums: List[int], raw_weights: Dict[int, float]) -> List[float]:
    ws = [max(0.0, float(raw_weights.get(n, 0.0))) for n in nums]
    s = sum(ws)
    if s <= 0.0:
        return [1.0 for _ in ws]
    return [w / s for w in ws]


def weighted_sample_without_replacement(nums: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
    """
    Amostragem ponderada simples (repetindo escolhas e removendo escolhido).
    Para Lotofácil (25 itens) é suficiente e estável.
    """
    if k <= 0:
        return []
    if k > len(nums):
        raise ValueError("k maior que o tamanho da lista.")

    chosen: List[int] = []
    pool_nums = nums[:]
    pool_w = weights[:]

    for _ in range(k):
        # normaliza
        s = sum(pool_w)
        if s <= 0:
            # se zerar, volta para uniforme
            pool_w = [1.0 for _ in pool_w]
            s = float(len(pool_w))

        r = rng.uniform(0, s)
        acc = 0.0
        idx = 0
        for i, w in enumerate(pool_w):
            acc += w
            if r <= acc:
                idx = i
                break
        chosen.append(pool_nums.pop(idx))
        pool_w.pop(idx)

    return chosen


def build_number_weights(stats: Stats, alpha: float = 0.9, mix_uniform: float = 0.20) -> Dict[int, float]:
    """
    Constrói pesos por dezena a partir do percentual histórico com suavização:
      base = (pct + eps)^alpha
      final = (1-mix)*base_norm + mix*uniform
    """
    eps = 1e-6
    base = {}
    for d in range(1, 26):
        base[d] = (stats.freq_pct[d] + eps) ** alpha

    # normaliza base
    total = sum(base.values())
    if total <= 0:
        total = 1.0
    base_norm = {d: base[d] / total for d in base}

    uni = 1.0 / 25.0
    out = {}
    for d in range(1, 26):
        out[d] = (1.0 - mix_uniform) * base_norm[d] + mix_uniform * uni
    return out


def gerar_aposta(stats: Stats, rng: random.Random, alpha: float, mix_uniform: float) -> List[int]:
    weights = build_number_weights(stats, alpha=alpha, mix_uniform=mix_uniform)

    # Escolhe k_pares conforme histórico
    k_pares = sample_k_from_distribution(stats.dist_pares_por_concurso, rng)
    k_pares = max(0, min(15, k_pares))
    k_impares = 15 - k_pares

    evens = [d for d in range(1, 26) if d % 2 == 0]
    odds = [d for d in range(1, 26) if d % 2 == 1]

    w_evens = normalize_weights(evens, weights)
    w_odds = normalize_weights(odds, weights)

    chosen_evens = weighted_sample_without_replacement(evens, w_evens, k_pares, rng)
    chosen_odds = weighted_sample_without_replacement(odds, w_odds, k_impares, rng)

    aposta = sorted(chosen_evens + chosen_odds)
    if len(aposta) != 15 or len(set(aposta)) != 15:
        raise RuntimeError("Falha interna: aposta inválida gerada (duplicatas ou tamanho incorreto).")
    return aposta


def eh_valida_mask(cand_mask: int, historico_masks: List[int]) -> bool:
    # inválida se existir interseção >= 14
    for m in historico_masks:
        if (cand_mask & m).bit_count() >= 14:
            return False
    return True


def gerar_apostas(
    n: int,
    stats: Stats,
    historico_masks: List[int],
    rng: random.Random,
    max_attempts: int = 200000,
    alpha: float = 0.9,
    mix_uniform: float = 0.20,
) -> List[List[int]]:
    """
    Gera N apostas válidas. Ajusta levemente a mistura uniforme se ficar difícil.
    """
    if n <= 0:
        return []

    apostas: List[List[int]] = []
    seen_masks: set[int] = set()

    attempts = 0
    local_mix = float(mix_uniform)

    while len(apostas) < n and attempts < max_attempts:
        attempts += 1
        aposta = gerar_aposta(stats, rng, alpha=alpha, mix_uniform=local_mix)
        cand_mask = dezenas_para_mask(aposta)

        if cand_mask in seen_masks:
            continue
        if not eh_valida_mask(cand_mask, historico_masks):
            # se estiver rejeitando demais, aumenta mistura uniforme (mais diversidade)
            if attempts % 5000 == 0:
                local_mix = min(0.60, local_mix + 0.05)
            continue

        seen_masks.add(cand_mask)
        apostas.append(aposta)

    if len(apostas) < n:
        raise RuntimeError(
            f"Não foi possível gerar {n} apostas válidas dentro de {max_attempts} tentativas. "
            "Tente aumentar '--max-attempts' ou alterar '--mix-uniform'/'--alpha'."
        )

    return apostas


def format_aposta(aposta: List[int]) -> str:
    return ", ".join(f"{d:02d}" for d in sorted(aposta))


# ----------------------------
# CLI / Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gerador de apostas Lotofácil com validação 14/15 e pesos históricos.")
    p.add_argument("--n", type=int, required=True, help="Quantidade de apostas a gerar.")
    p.add_argument("--seed", type=int, default=None, help="Seed para reprodutibilidade.")
    p.add_argument("--cache-dir", type=str, default="./dados", help="Diretório para cache do histórico.")
    p.add_argument("--saida", type=str, default=None, help="Arquivo para salvar as apostas geradas.")
    p.add_argument("--no-update", action="store_true", help="Não atualizar histórico via internet; usar apenas cache local.")
    p.add_argument("--no-tabelas", action="store_true", help="Não imprimir as tabelas (dezenas e par/ímpar).")
    p.add_argument("--max-attempts", type=int, default=200000, help="Máximo de tentativas para gerar as apostas válidas.")
    p.add_argument("--alpha", type=float, default=0.9, help="Expoente (suavização) para pesos de dezenas (ex.: 0.7..1.2).")
    p.add_argument("--mix-uniform", type=float, default=0.20, help="Mistura com distribuição uniforme (0..0.60 recomendado).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    cache_dir = Path(args.cache_dir)
    historico = carregar_historico(cache_dir, atualizar=(not args.no_update))

    if not historico:
        print("Histórico vazio. Não é possível continuar.", file=sys.stderr)
        return 2

    ultimo_concurso = max(c["concurso"] for c in historico)
    print(f"[INFO] Histórico carregado: {len(historico)} concursos | último concurso: {ultimo_concurso}", file=sys.stderr)

    stats = calcular_stats(historico)

    if not args.no_tabelas:
        print(imprimir_tabela_dezenas(stats))
        print()
        print(imprimir_tabela_par_impar(stats))
        print()

    historico_masks = preparar_masks_historico(historico)

    apostas = gerar_apostas(
        n=args.n,
        stats=stats,
        historico_masks=historico_masks,
        rng=rng,
        max_attempts=args.max_attempts,
        alpha=args.alpha,
        mix_uniform=max(0.0, min(0.60, args.mix_uniform)),
    )

    linhas = []
    for i, a in enumerate(apostas, start=1):
        linhas.append(f"Aposta {i}: {format_aposta(a)}")

    output = "\n".join(linhas)
    print(output)

    if args.saida:
        out_path = Path(args.saida)
        out_path.write_text(output + "\n", encoding="utf-8")
        print(f"[INFO] Apostas salvas em: {out_path.resolve()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
