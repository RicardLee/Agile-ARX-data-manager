import json
import glob
import requests
import sys
from datetime import datetime

def read_json_files(folder_path):
    """Read all JSON files from the specified folder."""
    json_files = glob.glob(f"{folder_path}/*.json")
    data = []
    for file in json_files:
        try:
            with open(file, 'r') as f:
                json_data = json.load(f)
                data.append(json_data)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return data

def generate_report(data):
    """Generate a report from the JSON data."""
    if not data:
        return "No data available for the report."

    total_runtime = 0
    device_count = len(data)
    statuses = []
    report_lines = [f"# Device Runtime Report - {datetime.now().strftime('%Y-%m-%d')}\n"]

    for entry in data:
        device_id = entry.get('device_id', 'Unknown')
        runtime = entry.get('runtime_hours', 0)
        status = entry.get('status', 'Unknown')
        date = entry.get('date', 'Unknown')

        total_runtime += runtime
        statuses.append(status)
        report_lines.append(f"- **Device ID**: {device_id}, **Date**: {date}, **Runtime**: {runtime} hours, **Status**: {status}")

    avg_runtime = total_runtime / device_count if device_count > 0 else 0
    operational_count = sum(1 for s in statuses if s.lower() == 'operational')
    
    report_lines.append("\n## Summary")
    report_lines.append(f"- **Total Devices**: {device_count}")
    report_lines.append(f"- **Total Runtime**: {total_runtime:.2f} hours")
    report_lines.append(f"- **Average Runtime per Device**: {avg_runtime:.2f} hours")
    report_lines.append(f"- **Operational Devices**: {operational_count}/{device_count}")

    return "\n".join(report_lines)

def send_to_feishu(report, webhook_url):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "msg_type": "text",
        "content": {
            "text": report
        }
    }
    try:
        response = requests.post(webhook_url, headers=headers, json=payload)
        print(f"Response Status: {response.status_code}")
        print(f"Response Text: {response.text}")  # 添加详细输出
        if response.status_code == 200:
            print("Report sent to Feishu successfully.")
        else:
            print(f"Failed to send report to Feishu: {response.text}")
    except Exception as e:
        print(f"Error sending to Feishu: {e}")

def main():
    folder_path = "data"  # Folder containing JSON files
    webhook_url = sys.argv[1] if len(sys.argv) > 1 else ""
    
    if not webhook_url:
        print("Error: Feishu webhook URL not provided.")
        sys.exit(1)

    # Read JSON files
    data = read_json_files(folder_path)
    
    # Generate report
    report = generate_report(data)
    print("Generated Report:\n", report)
    
    # Send to Feishu
    send_to_feishu(report, webhook_url)

if __name__ == "__main__":
    main()
