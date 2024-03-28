import os
import base64
from apscheduler.schedulers.blocking import BlockingScheduler
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import re

# Set the path to watch
WATCH_DIRECTORY = '/mnt/e/13_Jasper_diffused_samples/training/output/vids'
# Service account file
SERVICE_ACCOUNT_FILE = '/home/wisley/custom_diffusers_library/src/diffusers/jasper/google_service_account.json'
# Email details
TO_EMAIL = 'jsvanleuven@gmail.com'
FROM_EMAIL = 'email-videos@iconic-iridium-388806.iam.gserviceaccount.com'
SUBJECT = 'New video available'

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            send_email(event.src_path)

def send_email(file_path):
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/gmail.send'])
    service = build('gmail', 'v1', credentials=credentials)

    message = MIMEMultipart()
    message['to'] = TO_EMAIL
    message['from'] = FROM_EMAIL
    message['subject'] = SUBJECT

    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(file_path, "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file_path))
    message.attach(part)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    send_message = {'raw': raw_message}
    service.users().messages().send(userId="email-videos@iconic-iridium-388806.iam.gserviceaccount.com", body=send_message).execute()

def get_latest_file():
    files = [f for f in os.listdir(WATCH_DIRECTORY) if f.startswith('videojap_') and f.endswith('.avi')]
    if not files:
        return None
    # Extract the numerical part of the filename and sort
    files.sort(key=lambda f: int(re.search(r'(\d+)', f).group()))
    # Return the path of the latest file
    return os.path.join(WATCH_DIRECTORY, files[-1])

def check_for_new_files_and_send_emails():
    latest_file = get_latest_file()
    if latest_file:
        send_email(latest_file)

if __name__ == "__main__":
    check_for_new_files_and_send_emails()
    # print("we are running")
    # scheduler = BlockingScheduler()
    # # Schedule the check to run every 30 minutes
    # scheduler.add_job(check_for_new_files_and_send_emails, 'interval', minutes=1)
    # scheduler.start()