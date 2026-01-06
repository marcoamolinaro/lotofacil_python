import os
import sys
import random
import logging
import datetime as dt
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional

import requests
import pandas as pd

# ---------------------------------------------------------
# CONFIGURAÇÃO DE LOG
# ---------------------------------------------------------

LOG_LEVEL = logging.INFO  # altere para logging.DEBUG para mais detalhe

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ---------------------------------------------------------

CSV_PATH = "lotofacil_resultados.csv"
API_BASE_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
REQUEST_TIMEOUT = 10  # segundos


# ---------------------------------------------------------
# FUNÇÕES AUXILIARES - ACESSO À API
# ---------------------------------------------------------


def parse_data_apuracao(data_str: str) -> dt.date:
    """Converte data no formato 'dd/MM/yyyy' (API CAIXA) para date."""
    return dt.datetime.strptime(data_str, "%d/%m/%Y").date()


def fetch_concurso(numero: int) -> Dict:
    """Busca dados de um concurso específico da Lotofácil na API da CAIXA."""
    logger.info(f"[API] Buscando dados do concurso {numero}...")
    resp = requests.get(
        API_BASE_URL,
        params={"concurso": numero},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    logger.debug(f"[API] Resposta bruta do concurso {numero}: {resp.text[:200]}...")
    logger.info(f"[API] Concurso {numero} recebido com sucesso.")
    return resp.json()


def fetch_ultimo_concurso() -> Dict:
    """Busca o último concurso disponível da Lotofácil na API da CAIXA."""
    logger.info("[API] Buscando último concurso disponível da Lotofácil...")
    resp = requests.get(API_BASE_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    logger.debug(f"[API] Resposta bruta último concurso: {resp.text[:200]}...")
    logger.info("[API] Último concurso recebido com sucesso.")
    return resp.json()


def obter_data_minima_disponivel() -> Optional[dt.date]:
    """Obtém a data do primeiro concurso disponível (concurso 1) na API.

    Retorna None se não for possível consultar.
    """
    try:
        logger.info("[API] Consultando data mínima (concurso 1)...")
        dados = fetch_concurso(1)
        data_min = parse_data_apuracao(dados["dataApuracao"])  # type: ignore[index]
        logger.info(f"[API] Data mínima disponível: {data_min}")
        return data_min
    except Exception as e:
        logger.warning(f"[API] Não foi possível obter a data mínima: {e}")
        return None


def concurso_json_para_linha(concurso_json: Dict) -> Dict:
    """
    Converte o JSON do concurso em um dict no formato:
    {
        'concurso': int,
        'data': 'yyyy-mm-dd',
        'd1'...'d15': int
    }
    """
    numero = int(concurso_json["numero"])
    logger.debug(f"[PARSE] Convertendo dados do concurso {numero}...")
    data = parse_data_apuracao(concurso_json["dataApuracao"])
    dezenas_str = concurso_json.get("listaDezenas") or []
    dezenas = sorted(int(d) for d in dezenas_str)

    if len(dezenas) != 15:
        raise ValueError(f"Concurso {numero} não possui 15 dezenas.")

    linha = {
        "concurso": numero,
        "data": data.isoformat(),
    }
    for i, dez in enumerate(dezenas, start=1):
        linha[f"d{i}"] = dez

    logger.debug(f"[PARSE] Concurso {numero} convertido: {linha}")
    return linha


# ---------------------------------------------------------
# CSV / DATAFRAME - CARREGAR E ATUALIZAR
# ---------------------------------------------------------


def carregar_resultados_csv(caminho: str) -> pd.DataFrame:
    """Carrega o CSV se existir, caso contrário retorna DF vazio no formato correto."""
    if not os.path.exists(caminho):
        logger.info(f"[CSV] Arquivo {caminho} não encontrado. Criando dataframe vazio.")
        colunas = ["concurso", "data"] + [f"d{i}" for i in range(1, 16)]
        return pd.DataFrame(columns=colunas)

    logger.info(f"[CSV] Carregando dados do arquivo {caminho}...")
    df = pd.read_csv(caminho)
    logger.info(f"[CSV] {len(df)} registros carregados do CSV.")

    df["concurso"] = df["concurso"].astype(int)
    df["data"] = pd.to_datetime(df["data"]).dt.date
    for i in range(1, 16):
        df[f"d{i}"] = df[f"d{i}"].astype(int)
    logger.debug("[CSV] Conversão de tipos concluída.")
    return df


def salvar_resultados_csv(df: pd.DataFrame, caminho: str) -> None:
    """Salva o DF inteiro no CSV (reescreve o arquivo)."""
    logger.info(f"[CSV] Salvando {len(df)} registros no arquivo {caminho} (rewrite)...")
    df.to_csv(caminho, index=False)
    logger.info("[CSV] Salvamento concluído.")


def anexar_resultados_csv(df_novos: pd.DataFrame, caminho: str) -> None:
    """Anexa somente as novas linhas ao CSV existente, sem apagar o conteúdo anterior.

    Se o arquivo não existir, cria com cabeçalho. Se existir, escreve em modo append
    sem cabeçalho, preservando a ordem de colunas: concurso, data, d1..d15.
    """
    colunas = ["concurso", "data"] + [f"d{i}" for i in range(1, 16)]
    df_out = df_novos[colunas].copy()

    arquivo_existe = os.path.exists(caminho)
    if not arquivo_existe:
        logger.info(f"[CSV] Criando arquivo {caminho} com {len(df_out)} novos registros...")
        df_out.to_csv(caminho, index=False, mode="w", header=True)
    else:
        logger.info(f"[CSV] Anexando {len(df_out)} novos registros em {caminho}...")
        df_out.to_csv(caminho, index=False, mode="a", header=False)
    logger.info("[CSV] Operação de append concluída.")


def atualizar_resultados_ate_data(
    data_limite: dt.date, caminho_csv: str = CSV_PATH
) -> pd.DataFrame:
    """
    Garante que o CSV tenha todos os concursos da Lotofácil
    desde o 1 até os concursos com data <= data_limite.
    Faz atualização incremental.
    """
    logger.info(f"[ATUALIZAÇÃO] Iniciando atualização até a data {data_limite}...")
    df = carregar_resultados_csv(caminho_csv)

    if df.empty:
        ultimo_concurso_existente = 0
        logger.info("[ATUALIZAÇÃO] Nenhum dado local encontrado. Download completo necessário.")
    else:
        ultimo_concurso_existente = int(df["concurso"].max())
        logger.info(
            f"[ATUALIZAÇÃO] Dados locais até o concurso {ultimo_concurso_existente}."
        )

    try:
        ultimo_json = fetch_ultimo_concurso()
    except Exception as e:
        logger.error(f"[ERRO] Falha ao consultar último concurso na API: {e}")
        if df.empty:
            logger.critical("[ERRO] Sem dados locais e sem acesso à API. Encerrando.")
            sys.exit(1)
        else:
            logger.warning("[AVISO] Usando apenas dados locais, sem atualização.")
            return df

    numero_max_api = int(ultimo_json["numero"])
    logger.info(f"[ATUALIZAÇÃO] Último concurso na API: {numero_max_api}.")

    proximo = ultimo_concurso_existente + 1
    if proximo > numero_max_api:
        logger.info("[ATUALIZAÇÃO] CSV já está atualizado em relação à API.")
        df = df[df["data"] <= data_limite].copy()
        logger.info(
            f"[ATUALIZAÇÃO] Após filtro por data, {len(df)} concursos permanecem."
        )
        return df

    logger.info(
        f"[ATUALIZAÇÃO] Baixando concursos de {proximo} até {numero_max_api}, "
        f"limitado pela data {data_limite}."
    )

    novas_linhas = []
    for n in range(proximo, numero_max_api + 1):
        logger.debug(f"[LOOP-API] Processando concurso {n}...")
        try:
            dados = fetch_concurso(n)
        except Exception as e:
            logger.error(f"[ERRO] Erro ao buscar concurso {n}: {e}. Interrompendo atualização.")
            break

        data_concurso = parse_data_apuracao(dados["dataApuracao"])
        logger.debug(f"[LOOP-API] Concurso {n} tem data {data_concurso}.")
        if data_concurso > data_limite:
            logger.info(
                f"[ATUALIZAÇÃO] Data do concurso {n} ({data_concurso}) "
                f"é maior que {data_limite}. Parando loop de atualização."
            )
            break

        try:
            linha = concurso_json_para_linha(dados)
        except Exception as e:
            logger.error(f"[ERRO] Concurso {n} ignorado por erro de formato: {e}")
            continue

        novas_linhas.append(linha)
        logger.debug(f"[LOOP-API] Concurso {n} adicionado à lista de novos registros.")

    logger.info(f"[ATUALIZAÇÃO] Total de novos concursos obtidos: {len(novas_linhas)}.")

    if novas_linhas:
        df_novos = pd.DataFrame(novas_linhas)
        df_novos["data"] = pd.to_datetime(df_novos["data"]).dt.date
        logger.info("[ATUALIZAÇÃO] Concatenando novos concursos ao dataframe existente (memória)...")
        df = pd.concat([df, df_novos], ignore_index=True)
        df = df.drop_duplicates(subset=["concurso"]).sort_values("concurso")
        logger.info(f"[ATUALIZAÇÃO] Dataframe agora possui {len(df)} concursos.")

        # Persistência: se já havia dados locais, apenas anexar novas linhas para não apagar o arquivo.
        if ultimo_concurso_existente > 0 and os.path.exists(caminho_csv):
            anexar_resultados_csv(df_novos, caminho_csv)
        else:
            # Primeira criação do arquivo: salva o conjunto completo disponível até a data limite.
            salvar_resultados_csv(df, caminho_csv)
    else:
        logger.info("[ATUALIZAÇÃO] Nenhum concurso novo para adicionar ao CSV.")

    df = df[df["data"] <= data_limite].copy()
    logger.info(
        f"[ATUALIZAÇÃO] Após aplicar filtro por data, {len(df)} concursos permanecem."
    )
    logger.info("[ATUALIZAÇÃO] Concluída.")
    return df


# ---------------------------------------------------------
# GERAÇÃO DAS TABELAS ESTATÍSTICAS
# ---------------------------------------------------------


def gerar_L1(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[L1] Iniciando geração da Tabela L1...")
    col_dezenas = [f"d{i}" for i in range(1, 16)]
    cont = Counter()
    total_dezenas = len(df) * len(col_dezenas)
    logger.info(f"[L1] Total de concursos: {len(df)}, total de dezenas: {total_dezenas}.")

    for idx, (_, row) in enumerate(df[col_dezenas].iterrows(), start=1):
        dezenas = [int(v) for v in row.values]
        cont.update(dezenas)
        if idx % 200 == 0:
            logger.debug(f"[L1] Processando linha {idx}/{len(df)}...")

    linhas = []
    for dez in range(1, 26):
        freq = cont.get(dez, 0)
        pct = (freq / total_dezenas * 100) if total_dezenas > 0 else 0.0
        linhas.append(
            {
                "Dezena": dez,
                "Frequencia": freq,
                "% do total de dezenas": round(pct, 3),
            }
        )
    logger.debug("[L1] Construindo dataframe da L1...")
    l1 = pd.DataFrame(linhas)
    l1 = l1.sort_values(by=["Frequencia", "Dezena"], ascending=[False, True]).reset_index(
        drop=True
    )
    logger.info("[L1] Tabela L1 gerada com sucesso.")
    return l1


def gerar_distribuicao_par_impar(df: pd.DataFrame, label: str) -> pd.DataFrame:
    logger.info(f"[{label}] Iniciando geração de distribuição pares/ímpares...")
    col_dezenas = [f"d{i}" for i in range(1, 16)]
    dist = defaultdict(int)
    total_concursos = 0

    for idx, (_, row) in enumerate(df[col_dezenas].iterrows(), start=1):
        dezenas = [int(v) for v in row.values]
        pares = sum(1 for d in dezenas if d % 2 == 0)
        impares = len(dezenas) - pares
        dist[(pares, impares)] += 1
        total_concursos += 1
        if idx % 200 == 0:
            logger.debug(f"[{label}] Processando concurso {idx}/{len(df)}...")

    linhas = []
    for (pares, impares), freq in sorted(
        dist.items(), key=lambda x: (-x[1], x[0][0])
    ):
        pct = (freq / total_concursos * 100) if total_concursos > 0 else 0.0
        linhas.append(
            {
                "Pares": pares,
                "Impares": impares,
                "FrequenciaConcursos": freq,
                "% de concursos": round(pct, 3),
            }
        )

    logger.debug(f"[{label}] Construindo dataframe da distribuição...")
    df_out = pd.DataFrame(linhas)
    logger.info(f"[{label}] Distribuição pares/ímpares gerada com sucesso.")
    return df_out


def gerar_L2(df: pd.DataFrame) -> pd.DataFrame:
    return gerar_distribuicao_par_impar(df, "L2")


def gerar_L3(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[L3] Filtrando concursos com final 0...")
    df_final_zero = df[df["concurso"] % 10 == 0].copy()
    logger.info(f"[L3] Total de concursos com final 0: {len(df_final_zero)}.")
    if df_final_zero.empty:
        logger.warning("[L3] Nenhum concurso com final 0 para processar.")
        return pd.DataFrame(columns=["Pares", "Impares", "FrequenciaConcursos", "% de concursos"])
    return gerar_distribuicao_par_impar(df_final_zero, "L3")


def gerar_L4(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[L4] Filtrando concursos com final 0 para cálculo de L4...")
    df_final_zero = df[df["concurso"] % 10 == 0].copy()
    logger.info(f"[L4] Total de concursos com final 0: {len(df_final_zero)}.")
    if df_final_zero.empty:
        logger.warning("[L4] Nenhum concurso com final 0 disponível.")
        return pd.DataFrame(columns=["Dezena", "Frequencia", "% do total de dezenas"])
    return gerar_L1(df_final_zero)


# ---------------------------------------------------------
# GERAÇÃO DE APOSTAS
# ---------------------------------------------------------


def construir_pesos_dezenas(
    L1: Optional[pd.DataFrame],
    L4: Optional[pd.DataFrame],
    usar_L1: bool,
    usar_L4: bool,
) -> Dict[int, float]:
    logger.info("[PESOS] Construindo pesos das dezenas (L1/L4)...")
    pesos = {d: 1.0 for d in range(1, 26)}

    if usar_L1 and L1 is not None and not L1.empty:
        logger.info("[PESOS] Aplicando pesos a partir da L1...")
        max_freq = L1["Frequencia"].max() or 1
        freq_map = dict(zip(L1["Dezena"], L1["Frequencia"]))
        for d in range(1, 26):
            norm = freq_map.get(d, 0) / max_freq
            pesos[d] += norm

    if usar_L4 and L4 is not None and not L4.empty:
        logger.info("[PESOS] Aplicando pesos a partir da L4...")
        max_freq = L4["Frequencia"].max() or 1
        freq_map = dict(zip(L4["Dezena"], L4["Frequencia"]))
        for d in range(1, 26):
            norm = freq_map.get(d, 0) / max_freq
            pesos[d] += norm

    logger.debug(f"[PESOS] Pesos finais das dezenas: {pesos}")
    logger.info("[PESOS] Construção dos pesos de dezenas concluída.")
    return pesos


def construir_pesos_padroes_paridade(
    L2: Optional[pd.DataFrame], L3: Optional[pd.DataFrame], usar_L2: bool, usar_L3: bool
) -> Dict[Tuple[int, int], float]:
    logger.info("[PESOS] Construindo pesos dos padrões de paridade (L2/L3)...")
    padroes: Dict[Tuple[int, int], float] = defaultdict(float)

    if usar_L2 and L2 is not None and not L2.empty:
        logger.info("[PESOS] Incorporando padrões a partir da L2...")
        for _, row in L2.iterrows():
            p = int(row["Pares"])
            i = int(row["Impares"])
            freq = float(row["FrequenciaConcursos"])
            padroes[(p, i)] += freq

    if usar_L3 and L3 is not None and not L3.empty:
        logger.info("[PESOS] Incorporando padrões a partir da L3...")
        for _, row in L3.iterrows():
            p = int(row["Pares"])
            i = int(row["Impares"])
            freq = float(row["FrequenciaConcursos"])
            padroes[(p, i)] += freq

    logger.debug(f"[PESOS] Pesos finais de padrões: {padroes}")
    logger.info("[PESOS] Construção dos pesos de padrões de paridade concluída.")
    return padroes


def escolher_padrao_paridade(
    padroes_peso: Dict[Tuple[int, int], float],
    tamanho_aposta: int,
    dezenas_fixas: Set[int],
) -> Optional[Tuple[int, int]]:
    logger.info("[PADRÃO] Escolhendo padrão de paridade (se aplicável)...")
    if not padroes_peso or tamanho_aposta != 15:
        logger.info("[PADRÃO] Não será usado padrão de paridade (sem pesos ou tamanho != 15).")
        return None

    fixos_pares = sum(1 for d in dezenas_fixas if d % 2 == 0)
    fixos_impares = len(dezenas_fixas) - fixos_pares

    candidatos = []
    pesos = []
    for (p, i), w in padroes_peso.items():
        if p + i != 15:
            continue
        if p >= fixos_pares and i >= fixos_impares:
            candidatos.append((p, i))
            pesos.append(w)

    if not candidatos:
        logger.warning("[PADRÃO] Nenhum padrão compatível com as dezenas fixas.")
        return None

    padrao_escolhido = random.choices(candidatos, weights=pesos, k=1)[0]
    logger.info(f"[PADRÃO] Padrão escolhido: {padrao_escolhido[0]} pares, {padrao_escolhido[1]} ímpares.")
    return padrao_escolhido


def amostrar_sem_reposicao_ponderada(
    candidatos: List[int], pesos: Dict[int, float], k: int
) -> List[int]:
    escolhidos = []
    disponiveis = list(candidatos)

    for passo in range(1, min(k, len(disponiveis)) + 1):
        w = [pesos.get(d, 1.0) for d in disponiveis]
        total = sum(w) or 1.0
        probs = [x / total for x in w]
        escolhido = random.choices(disponiveis, weights=probs, k=1)[0]
        escolhidos.append(escolhido)
        disponiveis.remove(escolhido)
        logger.debug(f"[AMOSTRAGEM] Passo {passo}: escolhido {escolhido}.")

    return escolhidos


def gerar_uma_aposta(
    tamanho_aposta: int,
    dezenas_fixas: Set[int],
    pesos_dezenas: Dict[int, float],
    padroes_paridade_peso: Dict[Tuple[int, int], float],
) -> List[int]:
    logger.info("[APOSTA] Gerando uma nova aposta...")
    if tamanho_aposta < len(dezenas_fixas):
        raise ValueError(
            f"Tamanho da aposta ({tamanho_aposta}) menor que qtd de dezenas fixas ({len(dezenas_fixas)})."
        )

    padrao = escolher_padrao_paridade(
        padroes_paridade_peso, tamanho_aposta, dezenas_fixas
    )

    numeros_escolhidos = set(dezenas_fixas)
    logger.debug(f"[APOSTA] Dezenas fixas incluídas: {sorted(numeros_escolhidos)}")
    disponiveis = [d for d in range(1, 26) if d not in numeros_escolhidos]

    if padrao is not None:
        alvo_pares, alvo_impares = padrao
        fixos_pares = sum(1 for d in dezenas_fixas if d % 2 == 0)
        fixos_impares = len(dezenas_fixas) - fixos_pares

        restantes_pares = max(alvo_pares - fixos_pares, 0)
        restantes_impares = max(alvo_impares - fixos_impares, 0)

        falta_total = tamanho_aposta - len(numeros_escolhidos)
        if restantes_pares + restantes_impares > falta_total:
            excedente = restantes_pares + restantes_impares - falta_total
            if restantes_impares >= excedente:
                restantes_impares -= excedente
            else:
                excedente -= restantes_impares
                restantes_impares = 0
                restantes_pares = max(restantes_pares - excedente, 0)

        logger.debug(
            f"[APOSTA] Vamos tentar sortear {restantes_pares} pares e "
            f"{restantes_impares} ímpares adicionais."
        )

        pares_cand = [d for d in disponiveis if d % 2 == 0]
        impares_cand = [d for d in disponiveis if d % 2 != 0]

        escolhidos_pares = amostrar_sem_reposicao_ponderada(
            pares_cand, pesos_dezenas, restantes_pares
        )
        for d in escolhidos_pares:
            numeros_escolhidos.add(d)
            disponiveis.remove(d)

        escolhidos_impares = amostrar_sem_reposicao_ponderada(
            impares_cand, pesos_dezenas, restantes_impares
        )
        for d in escolhidos_impares:
            if d in disponiveis:
                numeros_escolhidos.add(d)
                disponiveis.remove(d)

    faltam = tamanho_aposta - len(numeros_escolhidos)
    logger.debug(f"[APOSTA] Ainda faltam {faltam} dezenas para completar a aposta.")
    if faltam > 0 and disponiveis:
        extras = amostrar_sem_reposicao_ponderada(disponiveis, pesos_dezenas, faltam)
        for d in extras:
            numeros_escolhidos.add(d)

    aposta = sorted(numeros_escolhidos)
    logger.debug(f"[APOSTA] Aposta gerada (antes de ajuste final): {aposta}")
    if len(aposta) > tamanho_aposta:
        aposta = aposta[:tamanho_aposta]
    elif len(aposta) < tamanho_aposta:
        for d in range(1, 26):
            if d not in aposta:
                aposta.append(d)
            if len(aposta) == tamanho_aposta:
                break
        aposta = sorted(aposta)

    logger.info(f"[APOSTA] Aposta final: {aposta}")
    return aposta


def gerar_apostas(
    qtd_apostas: int,
    tamanho_aposta: int,
    dezenas_fixas: Set[int],
    tabelas_usadas: Set[int],
    L1: pd.DataFrame,
    L2: pd.DataFrame,
    L3: pd.DataFrame,
    L4: pd.DataFrame,
) -> List[List[int]]:
    logger.info("[APOSTAS] Iniciando geração de apostas...")
    usar_L1 = 1 in tabelas_usadas
    usar_L2 = 2 in tabelas_usadas
    usar_L3 = 3 in tabelas_usadas
    usar_L4 = 4 in tabelas_usadas

    logger.info(f"[APOSTAS] Tabelas usadas: {sorted(tabelas_usadas)}")
    logger.info(f"[APOSTAS] Dezenas fixas: {sorted(dezenas_fixas)}")
    logger.info(f"[APOSTAS] Tamanho da aposta: {tamanho_aposta}")
    logger.info(f"[APOSTAS] Quantidade de apostas: {qtd_apostas}")

    pesos_dezenas = construir_pesos_dezenas(
        L1 if not L1.empty else None,
        L4 if not L4.empty else None,
        usar_L1,
        usar_L4,
    )

    padroes_paridade_peso = construir_pesos_padroes_paridade(
        L2 if not L2.empty else None,
        L3 if not L3.empty else None,
        usar_L2,
        usar_L3,
    )

    apostas = []
    for idx in range(1, qtd_apostas + 1):
        logger.info(f"[APOSTAS] Gerando aposta {idx}/{qtd_apostas}...")
        aposta = gerar_uma_aposta(
            tamanho_aposta,
            dezenas_fixas,
            pesos_dezenas,
            padroes_paridade_peso,
        )
        apostas.append(aposta)

    logger.info("[APOSTAS] Todas as apostas foram geradas.")
    return apostas


# ---------------------------------------------------------
# ENTRADA DO USUÁRIO
# ---------------------------------------------------------


def ler_data_limite() -> dt.date:
    s = input("Informe a data do último concurso (dd/mm/aaaa): ").strip()
    try:
        data = dt.datetime.strptime(s, "%d/%m/%Y").date()
        logger.info(f"[INPUT] Data limite informada: {data}")
        return data
    except ValueError:
        hoje = dt.date.today()
        logger.error(f"[ERRO INPUT] Data inválida. Usando hoje ({hoje}) como padrão.")
        return hoje


def ler_tabelas_usadas() -> Set[int]:
    s = input(
        "Quais tabelas usar para cruzamento? (1,2,3,4) ou vazio para nenhuma: "
    ).strip()
    if not s:
        logger.info("[INPUT] Nenhuma tabela será usada no cruzamento.")
        return set()
    partes = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    result = set()
    for p in partes:
        try:
            v = int(p)
            if v in {1, 2, 3, 4}:
                result.add(v)
            else:
                logger.error(f"[ERRO INPUT] Tabela inválida ignorada: {v}")
        except ValueError:
            logger.error(f"[ERRO INPUT] Valor inválido ignorado: {p}")
            continue
    logger.info(f"[INPUT] Tabelas selecionadas: {sorted(result)}")
    return result


def ler_dezenas_fixas() -> Set[int]:
    s = input(
        "Informe dezenas fixas (1-25) separadas por vírgula ou deixe vazio: "
    ).strip()
    if not s:
        logger.info("[INPUT] Nenhuma dezena fixa informada.")
        return set()
    partes = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    fixas = set()
    for p in partes:
        try:
            v = int(p)
            if 1 <= v <= 25:
                fixas.add(v)
            else:
                logger.error(f"[ERRO INPUT] Dezena fora do intervalo 1-25 ignorada: {v}")
        except ValueError:
            logger.error(f"[ERRO INPUT] Valor inválido ignorado: {p}")
            continue
    logger.info(f"[INPUT] Dezenas fixas: {sorted(fixas)}")
    return fixas


def ler_tamanho_aposta() -> int:
    s = input(
        "Quantidade de dezenas por aposta (0 ou vazio para 15, mínimo 15, máximo 20): "
    ).strip()
    if not s or s == "0":
        logger.info("[INPUT] Tamanho de aposta padrão selecionado: 15 dezenas.")
        return 15
    try:
        v = int(s)
    except ValueError:
        logger.error("[ERRO INPUT] Valor inválido. Usando 15 dezenas.")
        return 15
    if v < 15 or v > 20:
        logger.error("[ERRO INPUT] Tamanho fora do intervalo permitido (15-20). Usando 15.")
        return 15
    logger.info(f"[INPUT] Tamanho de aposta selecionado: {v} dezenas.")
    return v


def ler_qtd_apostas() -> int:
    s = input("Quantidade de apostas a serem geradas: ").strip()
    try:
        v = int(s)
        if v <= 0:
            raise ValueError()
        logger.info(f"[INPUT] Quantidade de apostas: {v}")
        return v
    except ValueError:
        logger.error("[ERRO INPUT] Valor inválido. Usando 5 apostas.")
        return 5


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    logger.info("===============================================")
    logger.info("   Análise Lotofácil - Tabelas L1, L2, L3, L4  ")
    logger.info("   Geração de Apostas com Base Estatística     ")
    logger.info("===============================================")

    try:
        logger.info("----------------- DADOS DE ENTRADA -----------------")
        data_limite = ler_data_limite()
        tabelas_usadas = ler_tabelas_usadas()
        dezenas_fixas = ler_dezenas_fixas()
        tamanho_aposta = ler_tamanho_aposta()
        qtd_apostas = ler_qtd_apostas()

        if len(dezenas_fixas) > tamanho_aposta:
            logger.error(
                f"[ERRO] Quantidade de dezenas fixas ({len(dezenas_fixas)}) "
                f"maior que o tamanho da aposta ({tamanho_aposta}). Encerrando."
            )
            return

        logger.info("[ETAPA] Atualizando/Carregando resultados até a data limite...")
        df = atualizar_resultados_ate_data(data_limite, CSV_PATH)
        if df.empty:
            # Fallback: se a data informada é anterior ao primeiro concurso, usar a data mínima disponível
            data_min = obter_data_minima_disponivel()
            if data_min and data_limite < data_min:
                logger.warning(
                    f"[AVISO] Nenhum concurso até {data_limite}. Primeira data disponível é {data_min}. "
                    f"Refazendo atualização usando {data_min}."
                )
                data_limite = data_min
                df = atualizar_resultados_ate_data(data_limite, CSV_PATH)

        if df.empty:
            logger.error("[ERRO] Nenhum concurso disponível até a data informada. Encerrando.")
            return

        logger.info("[ETAPA] Gerando tabelas estatísticas L1, L2, L3, L4...")
        L1 = gerar_L1(df)
        L2 = gerar_L2(df)
        L3 = gerar_L3(df)
        L4 = gerar_L4(df)

        logger.info("[ETAPA] Gerando apostas da Lotofácil...")
        apostas = gerar_apostas(
            qtd_apostas,
            tamanho_aposta,
            dezenas_fixas,
            tabelas_usadas,
            L1,
            L2,
            L3,
            L4,
        )

        print("\n==================== APOSTAS GERADAS ====================")
        for i, aposta in enumerate(apostas, start=1):
            dezenas_str = " ".join(f"{d:02d}" for d in sorted(aposta))
            print(f"Aposta {i:02d}: {dezenas_str}")

        print("\n------------------ TABELA L1 ------------------")
        print(L1.to_string(index=False))

        print("\n------------------ TABELA L2 ------------------")
        print(L2.to_string(index=False))

        print("\n------------------ TABELA L3 ------------------")
        if L3.empty:
            print("Nenhum concurso com final 0 encontrado até a data limite.")
        else:
            print(L3.to_string(index=False))

        print("\n------------------ TABELA L4 ------------------")
        if L4.empty:
            print("Nenhum concurso com final 0 encontrado até a data limite.")
        else:
            print(L4.to_string(index=False))

        logger.info("[FINAL] Processamento concluído com sucesso.")

    except Exception as e:
        logger.critical(
            "[ERRO FATAL] Ocorreu um erro inesperado durante a execução do programa.",
            exc_info=True,
        )
        print("\nO programa encontrou um erro inesperado e foi interrompido.")
        print(f"Detalhes: {e}")


if __name__ == "__main__":
    main()
