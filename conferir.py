import csv
import os


def ler_jogos_csv(nome_arquivo):
    jogos = []

    if not os.path.exists(nome_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {nome_arquivo}")

    with open(nome_arquivo, mode="r", encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.reader(arquivo)

        for numero_linha, linha in enumerate(leitor, start=1):
            dezenas = [campo.strip() for campo in linha if campo.strip() != ""]

            if not dezenas:
                continue

            try:
                jogo = set(map(int, dezenas))
            except ValueError:
                raise ValueError(
                    f"Linha {numero_linha} do arquivo '{nome_arquivo}' contém valor inválido."
                )

            jogos.append(jogo)

    if not jogos:
        raise ValueError(f"O arquivo '{nome_arquivo}' está vazio ou não possui jogos válidos.")

    return jogos


def conferir_apostas(apostas, resultados):
    linhas_saida = []
    linhas_saida.append("=== Conferência de Apostas ===")

    for i, aposta in enumerate(apostas, start=1):
        linhas_saida.append(f"\nAposta {i}: {sorted(aposta)}")

        for j, resultado in enumerate(resultados, start=1):
            acertos = sorted(aposta.intersection(resultado))
            quantidade_acertos = len(acertos)

            linhas_saida.append(f"  Resultado {j}: {sorted(resultado)}")
            linhas_saida.append(f"    Quantidade de acertos: {quantidade_acertos}")
            linhas_saida.append(f"    Dezenas acertadas: {acertos if acertos else 'Nenhuma'}")

    return linhas_saida


def salvar_saida_txt(nome_arquivo, linhas):
    with open(nome_arquivo, mode="w", encoding="utf-8") as arquivo:
        for linha in linhas:
            arquivo.write(linha + "\n")


def main():
    print("=== Conferidor de Apostas ===")

    arquivo_apostas = input("Informe o nome do arquivo CSV de apostas: ").strip()
    arquivo_resultados = input("Informe o nome do arquivo CSV de resultados: ").strip()
    arquivo_saida = input("Informe o nome do arquivo TXT de saída (ex: saida.txt): ").strip()

    try:
        apostas = ler_jogos_csv(arquivo_apostas)
        resultados = ler_jogos_csv(arquivo_resultados)

        linhas_saida = conferir_apostas(apostas, resultados)

        for linha in linhas_saida:
            print(linha)

        salvar_saida_txt(arquivo_saida, linhas_saida)

        print(f"\nSaída gravada com sucesso no arquivo: {arquivo_saida}")

    except Exception as e:
        print(f"\nErro: {e}")


if __name__ == "__main__":
    main()