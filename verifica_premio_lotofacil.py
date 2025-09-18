import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# 🔐 Configurações de e-mail
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_USER = 'marco.amolinaro@gmail.com'
EMAIL_PASSWORD = 'xcmr xwpe nqxd veng'
DESTINATARIO = 'molinaromarcoaurelio@gmail.com'


# 🎯 Função para enviar e-mail
def send_email(premio, concurso):
    mensagem = MIMEMultipart()
    mensagem['From'] = EMAIL_USER
    mensagem['To'] = DESTINATARIO
    mensagem['Subject'] = '🚨 Prêmio da Lotofácil Acima de 2 Milhões!'

    body = (f'O prêmio da Lotofácil do concurso {concurso} '
            f'está em R$ {premio:,.2f}!\n\n'
            f'Aproveite para apostar!')
    mensagem.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(mensagem)
        server.quit()
        print('✅ E-mail enviado com sucesso!')
    except Exception as e:
        print(f'❌ Erro ao enviar e-mail: {e}')
        
# Função para enviar mensagem por email
def send_msg_email():
    mensagem = MIMEMultipart()
    mensagem['From'] = EMAIL_USER
    mensagem['To'] = DESTINATARIO
    mensagem['Subject'] = '🚨 Prêmio da Lotofácil abaixo de 2 Milhões!'    
    body = (f'O prêmio da Lotofácil do concurso {concurso} não ultrapassou de 2 Milhões')
    mensagem.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(mensagem)
        server.quit()
        print('✅ E-mail enviado com sucesso!')
    except Exception as e:
        print(f'❌ Erro ao enviar e-mail: {e}')
        

# 🔍 Função para obter o prêmio da Lotofácil
def obter_premio_lotofacil():
    url = 'https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil'
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        dados = response.json()

        concurso = dados['numero'] + 1
        premio_estimado = dados['valorEstimadoProximoConcurso']

        print(f"🔎 Concurso: {concurso}")
        print(f"💰 Prêmio estimado: R$ {premio_estimado:,.2f}")

        return premio_estimado, concurso

    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao acessar a API: {e}")
        return None, None


# 🚦 Execução principal
if __name__ == "__main__":
    premio, concurso = obter_premio_lotofacil()

    if premio is not None:
        if premio > 2000000.00:
            send_email(premio, concurso)
        else:
            send_msg_email()
            print("ℹ️ O prêmio não ultrapassou R$ 2.000.000,00.")