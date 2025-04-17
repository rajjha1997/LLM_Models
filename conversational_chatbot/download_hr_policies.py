# The below code downloads HR policies from the specified URL and saves them in the specified output directory.
# It uses the wget command to recursively download HTML files.
# The subprocess module is used to run the command in the shell.
# The command is executed with error handling to ensure that any issues during the download process are reported.
# The function download_hr_policies takes an optional output_dir argument to specify where to save the downloaded files.
# The default output directory is "lanchain-docs".
# The URL points to a page that contains various HR policies.
# The command uses the -r option for recursive download and -A.html to accept only HTML files.
# The output directory is created if it doesn't exist.
# The function prints a success message if the download is successful and an error message if it fails.
# The downloaded files can be used for further processing or analysis.
import os
import subprocess

def download_hr_policies(output_dir="lanchain-docs"):
    url = "https://www.hrhelpboard.com/hr-policies.html"
    command = f"wget -r -A.html -P {output_dir} {url}"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print("✅ HR policy HTMLs downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Failed to download HTMLs:", e)

download_hr_policies()
