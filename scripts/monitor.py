#!/usr/bin/env python3
import os
import time
import socket
import smtplib
import pandas as pd
import json
from cryptography.fernet import Fernet
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from machine_learning_finance import plot_backtest_analysis, back_test_expert, make_inverse_env_for
import os
import urllib.request

proxy_server = None
proxy_port = None

def check_internet(host="http://google.com"):
    print("check_internet_without_proxy:")
    try:
        urllib.request.urlopen(host, timeout=3)
        return True
    except:
        return False


def check_internet_with_proxy(host="http://google.com"):
    global proxy_server
    global proxy_port

    if check_internet(host):
        return True
    else:
        print("check_internet_with_proxy:")
        proxy = os.getenv('MY_PROXY')
        port = os.getenv('MY_PORT')
        proxy_str = f'{proxy}:{port}'
        if proxy:
            proxy_handler = urllib.request.ProxyHandler({'http': proxy_str, 'https': proxy_str})
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
            try:
                print("trying: ", proxy, port)
                urllib.request.urlopen(host, timeout=3)
                proxy_server = proxy
                proxy_port = port
                return True
            except:
                return False
        else:
            return False


def send_email_with_attachments(to, subject, files):
    with open("./keys/key.key", "rb") as key_file:
        key = key_file.read()

    fernet = Fernet(key)

    with open("./keys/credentials.enc", "rb") as encrypted_file:
        encrypted_data = encrypted_file.read()

    decrypted_data = fernet.decrypt(encrypted_data)
    credentials = json.loads(decrypted_data.decode("utf-8"))

    from_email = credentials['username']
    password = credentials['password']

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to
    msg['Subject'] = subject

    for file in files:
        attachment = open(file, 'rb')
        base = MIMEBase('application', 'octet-stream')
        base.set_payload(attachment.read())
        encoders.encode_base64(base)
        base.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file)}')
        msg.attach(base)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to, msg.as_string())
    server.quit()


def main():
    # Read the CSV file with the symbols
    pairs = pd.read_csv('./lists/inverse_pairs.csv')

    # Wait for the internet connection
    while not check_internet_with_proxy():
        print("No internet connection. Waiting 5 minutes to check again.")
        time.sleep(300)

    # Variables to store files to be emailed
    png_files = []
    csv_files = []

    for index, row in pairs.iterrows():
        main_symbol, inverse_symbol = row['Main'], row['Inverse']
        png_file_name = f'backtests/{main_symbol}_and_{inverse_symbol}_backtest.png'
        csv_file_name = f'backtests/{main_symbol}_and_{inverse_symbol}_ledger.csv'

        print(f"scanning: {main_symbol} with {inverse_symbol}")

        env = make_inverse_env_for(main_symbol,
                                   inverse_symbol,
                                   1,
                                   365,
                                   cash=85000,
                                   prob_high=0.9,
                                   prob_low=0.1,
                                   proxy=proxy_server,
                                   port=proxy_port)
        print(f"back testing {main_symbol}")
        env = back_test_expert(env)

        last_row = env.timeseries.iloc[-1]

        print(f"Checking: {last_row}")

        if last_row["action"] != 0:
            print("****ACTION EXPECTED:", last_row)
            # Save the ledger as a CSV
            env.ledger.to_csv(csv_file_name, index=False)
            csv_files.append(csv_file_name)

            # Save the figure as a PNG
            plot_backtest_analysis(env.orig_timeseries, env.ledger, True, png_file_name)
            png_files.append(png_file_name)

    # If there are any files to be emailed, send the email
    if len(png_files) > 0:
        send_email_with_attachments('jmordetsky@gmail.com', 'Backtest Results', png_files + csv_files)


if __name__ == "__main__":
    main()
