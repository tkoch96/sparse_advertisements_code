"""Email a progress report with figure attachments to Tom (standing
request). Creds reused from ~/Documents/budgeter/emailer.py.
Usage: send_report.py <subject> <body_file> [attachment ...]"""
import smtplib
import sys
from email.message import EmailMessage

SENDER = "tomkoch123@gmail.com"
APP_PASSWORD = "tmeyvzdskvpaivmj"
DEST = "tomkoch123@gmail.com"


def main():
    subject, body_file = sys.argv[1], sys.argv[2]
    atts = sys.argv[3:]
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SENDER
    msg['To'] = DEST
    msg.set_content(open(body_file).read())
    for a in atts:
        with open(a, 'rb') as f:
            data = f.read()
        sub = 'png' if a.endswith('.png') else 'pdf'
        mt = 'image' if sub == 'png' else 'application'
        msg.add_attachment(data, maintype=mt, subtype=sub,
                           filename=a.split('/')[-1])
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
        s.login(SENDER, APP_PASSWORD)
        s.send_message(msg)
    print('sent:', subject)


if __name__ == '__main__':
    main()
