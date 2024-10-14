import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
smtp_server = 'smtp.office365.com'
smtp_port = 587
email_address = 'mukesh@adarone.com'
email_password = 'PASSSSSWORDDD'  # Use app password if 2FA is enabled
recipient_address = 'mukesh@adarone.com'

# Create the email content
subject = 'Test Email from Python'
body = 'This is a test email sent from a Python script!'

# Create a multipart email
msg = MIMEMultipart()
msg['From'] = email_address
msg['To'] = recipient_address
msg['Subject'] = subject

# Attach the email body to the email
msg.attach(MIMEText(body, 'plain'))

try:
    # Set up the server and send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Upgrade the connection to secure
        server.login(email_address, email_password)
        server.sendmail(email_address, recipient_address, msg.as_string())
        print("Email sent successfully!")
except Exception as e:
    print(f"Error: {e}")
