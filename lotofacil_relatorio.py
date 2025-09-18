
import sys
import time
import math
import argparse
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

CAIXA_BASE = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
HEADERS = {
    "Accept": "application/json; charset=utf-8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def fetch_concurso(numero: int, retries: int = 3, backoff: float = 1.5) -> Dict:
    """
    Baixa o JSON de um concurso da Lotofácil no portal de Loterias CAIXA.
    Retorna o dicionário JSON (ou lança exceção após esgotar retries).
    """
    print(f"Baixando concurso {numero} ...")
    url = f"{CAIXA_BASE}/{numero}"
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            # às vezes o portal usa 202/204 quando está atualizando
            if r.status_code in (202, 204, 429, 503):
                last_err = RuntimeError(f"HTTP {r.status_code} em {url}")
            else:
                r.raise_for_status()
        except Exception as e:
            last_err = e
        sleep_s = backoff ** i
        time.sleep(sleep_s)
    raise RuntimeError(f"Falha ao obter concurso {numero}: {last_err}")

def parse_dezenas(payload: Dict) -> Tuple[int, List[int]]:
    """
    Extrai (numero_do_concurso, lista_de_15_dezenas_int) do payload.
    O portal pode usar 'listaDezenas' ou 'dezenas' (strings).
    """
    print("Analisando dezenas ...")
    numero = payload.get("numero")
    dezenas_raw = payload.get("listaDezenas") or payload.get("dezenas") or []
    # normaliza para ints
    dezenas = []
    for d in dezenas_raw:
        try:
            dezenas.append(int(str(d).strip()))
        except:
            pass
    if len(dezenas) != 15:
        raise ValueError(f"Concurso {numero}: esperado 15 dezenas, obtido {len(dezenas)} ({dezenas_raw})")
    return numero, sorted(dezenas)

def baixar_intervalo(conc_ini: int, conc_fim: int, cache_csv: Path) -> pd.DataFrame:
    """
    Baixa (ou recarrega do CSV) os concursos no intervalo e devolve um DataFrame:
    colunas: Concurso, D1..D15
    """
    print(f"Verificando cache em {cache_csv} ...")
    if cache_csv.exists():
        df = pd.read_csv(cache_csv, sep=";", encoding="utf-8")
        df = df[(df["Concurso"] >= conc_ini) & (df["Concurso"] <= conc_fim)].copy()
        faltantes = set(range(conc_ini, conc_fim + 1)) - set(df["Concurso"].tolist())
    else:
        df = pd.DataFrame()
        faltantes = set(range(conc_ini, conc_fim + 1))

    print(f"Concursos faltantes no cache: {sorted(faltantes)}")
    registros = []
    for n in sorted(faltantes):
        payload = fetch_concurso(n)
        _, dezenas = parse_dezenas(payload)
        registros.append({"Concurso": n, **{f"D{i+1}": dezenas[i] for i in range(15)}})
        # breve pausa para evitar throttling
        time.sleep(0.15)

    print(f"Atualizando cache em {cache_csv} ...")
    if registros:
        df_novo = pd.DataFrame(registros)
        if df.empty:
            df = df_novo
        else:
            df = pd.concat([df, df_novo], ignore_index=True)
        df = df.sort_values("Concurso").reset_index(drop=True)
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, sep=";", index=False, encoding="utf-8")

    # sanidade
    cols = ["Concurso"] + [f"D{i}" for i in range(1, 16)]
    df = df[cols].sort_values("Concurso").reset_index(drop=True)
    return df

def freq_dezenas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna tabela com frequência de cada dezena (1..25), ordenada por dezena (crescente).
    """
    print("Calculando frequência das dezenas ...")
    todas = df[[f"D{i}" for i in range(1, 16)]].values.flatten()
    s = pd.Series(todas, dtype="int64")
    freq = s.value_counts().sort_index()
    # garante todas as dezenas 1..25
    freq = freq.reindex(range(1, 26), fill_value=0)
    out = pd.DataFrame({"Dezena": freq.index, "Frequencia": freq.values})
    out["% do total de dezenas"] = (out["Frequencia"] / s.size * 100).round(3)
    return out

def paridade_totais(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Conta totais absolutos de dezenas pares e ímpares no dataframe.
    """
    print("Calculando totais de paridade ...")  
    todas = pd.Series(df[[f"D{i}" for i in range(1, 16)]].values.flatten(), dtype="int64")
    pares = int((todas % 2 == 0).sum())
    impares = int((todas % 2 != 0).sum())
    return pares, impares

def distribuicao_paridade_por_concurso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a distribuição do número de dezenas pares (0..15) por concurso.
    Retorna uma tabela com colunas: ParesNoConcurso, FrequenciaConcursos, %
    (Ímpares = 15 - ParesNoConcurso)
    """
    print("Calculando distribuição de paridade por concurso ...")
    pares_por_concurso = []
    for _, row in df.iterrows():
        dezenas = [row[f"D{i}"] for i in range(1, 16)]
        qtd_pares = sum(1 for d in dezenas if d % 2 == 0)
        pares_por_concurso.append(qtd_pares)
    s = pd.Series(pares_por_concurso, dtype="int64")
    dist = s.value_counts().sort_index()
    out = pd.DataFrame({
        "ParesNoConcurso": dist.index,
        "ImparesNoConcurso": 15 - dist.index,
        "FrequenciaConcursos": dist.values
    })
    out["% de concursos"] = (out["FrequenciaConcursos"] / len(df) * 100).round(3)
    return out

def gerar_relatorio(conc_ini: int, conc_fim: int, saida_dir: Path) -> None:
    print(f'Inicio geração do relatório...')
    saida_dir.mkdir(parents=True, exist_ok=True)
    cache_csv = saida_dir / "lotofacil_resultados.csv"

    print(f"Baixando/atualizando concursos {conc_ini}..{conc_fim} …")
    df = baixar_intervalo(conc_ini, conc_fim, cache_csv)

    # 1) Dezenas mais sorteadas (ordem crescente da dezena)
    tabela_dezenas = freq_dezenas(df)

    # 2) Paridade no intervalo
    pares_total, impares_total = paridade_totais(df)
    resumo_paridade_geral = pd.DataFrame([{
        "Concursos": len(df),
        "Total_dezenas": len(df) * 15,
        "Pares": pares_total,
        "Impares": impares_total,
        "%Pares": round(pares_total / (len(df) * 15) * 100, 3),
        "%Impares": round(impares_total / (len(df) * 15) * 100, 3)
    }])

    dist_paridade_geral = distribuicao_paridade_por_concurso(df)

    # 3) Paridade nos concursos de final zero
    df_final0 = df[df["Concurso"] % 10 == 0].copy()
    if not df_final0.empty:
        p0, i0 = paridade_totais(df_final0)
        resumo_paridade_final0 = pd.DataFrame([{
            "Concursos_final0": len(df_final0),
            "Total_dezenas_final0": len(df_final0) * 15,
            "Pares_final0": p0,
            "Impares_final0": i0,
            "%Pares_final0": round(p0 / (len(df_final0) * 15) * 100, 3),
            "%Impares_final0": round(i0 / (len(df_final0) * 15) * 100, 3)
        }])
        dist_paridade_final0 = distribuicao_paridade_por_concurso(df_final0)
    else:
        resumo_paridade_final0 = pd.DataFrame([{
            "Concursos_final0": 0, "Total_dezenas_final0": 0,
            "Pares_final0": 0, "Impares_final0": 0,
            "%Pares_final0": 0.0, "%Impares_final0": 0.0
        }])
        dist_paridade_final0 = pd.DataFrame(columns=["ParesNoConcurso","ImparesNoConcurso","FrequenciaConcursos","% de concursos"])

    # salva Excel com abas
    xlsx_path = saida_dir / "relatorio_lotofacil.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xlw:
        tabela_dezenas.to_excel(xlw, sheet_name="Frequencia_Dezenas", index=False)
        resumo_paridade_geral.to_excel(xlw, sheet_name="Paridade_Geral_Resumo", index=False)
        dist_paridade_geral.to_excel(xlw, sheet_name="Paridade_Geral_Distrib", index=False)
        resumo_paridade_final0.to_excel(xlw, sheet_name="Paridade_Final0_Resumo", index=False)
        dist_paridade_final0.to_excel(xlw, sheet_name="Paridade_Final0_Distrib", index=False)
        # também salva o “raw”
        df.to_excel(xlw, sheet_name="Concursos_RAW", index=False)

    # salva CSV “raw” (já é salvo como cache, mas garantimos)
    df.to_csv(cache_csv, sep=";", index=False, encoding="utf-8")

    # imprime resumo no terminal
    print("\n===== RELATÓRIO LOTOFÁCIL =====")
    print(f"Concursos considerados: {conc_ini} até {conc_fim} (total: {len(df)})")
    print("\n1) Dezenas mais sorteadas (ordem crescente da dezena):")
    print(tabela_dezenas.to_string(index=False))

    print("\n2) Paridade no intervalo:")
    print(resumo_paridade_geral.to_string(index=False))

    print("\n   Distribuição (#pares por concurso) no intervalo:")
    print(dist_paridade_geral.to_string(index=False))

    print("\n3) Paridade apenas nos concursos de final 0:")
    print(resumo_paridade_final0.to_string(index=False))

    print("\n   Distribuição (#pares por concurso) — final 0:")
    if dist_paridade_final0.empty:
        print("(não há concursos final 0 no intervalo informado)")
    else:
        print(dist_paridade_final0.to_string(index=False))

    print(f"\nArquivos gerados em: {saida_dir.resolve()}")
    print(f"- {cache_csv.name} (CSV com os concursos)")
    print(f"- {xlsx_path.name} (relatório em Excel)")

def main():
    parser = argparse.ArgumentParser(
        description="Gera relatório Lotofácil: 1) dezenas mais sorteadas; 2) paridade geral; 3) paridade em finais 0."
    )
    parser.add_argument("concurso_inicial", type=int, help="Número inicial do concurso")
    parser.add_argument("concurso_final", type=int, help="Número final do concurso")
    parser.add_argument("--saida", type=str, default="saida_lotofacil", help="Diretório de saída (default: saida_lotofacil)")
    args = parser.parse_args()

    if args.concurso_inicial > args.concurso_final:
        print("Erro: concurso_inicial não pode ser maior que concurso_final.", file=sys.stderr)
        sys.exit(1)

    try:
        gerar_relatorio(args.concurso_inicial, args.concurso_final, Path(args.saida))
    except Exception as e:
        print(f"Falha ao gerar relatório: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
